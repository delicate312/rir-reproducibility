"""Recover first alerts from frozen caches and saved standalone decisions.

No model training, candidate selection, or threshold calibration. The original
scoring functions reconstruct the saved safe reference mappings. Every event
peak, hit and previously reported deadline must reproduce the saved record.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();root=args.project_root.resolve();out=args.output_root.resolve()
    sys.path.insert(0,str(root))
    from experiments.ego_immune_recognition.riskbench_metadata import standalone_rir_audit as a
    from experiments.ego_immune_recognition.riskbench_metadata import relation_geometry_rescue as geometry_module
    from experiments.ego_immune_recognition.riskbench_metadata.lazy_dataset import load_lazy_package
    d=a.d
    def relocated_load(index):
        payload=json.loads(Path(index).read_text())
        def relocate(s):
            return str(root/s.split('/Users/finn/ImmuneRecognition/',1)[1]) if '/Users/finn/ImmuneRecognition/' in s else s
        payload['cache_root']=relocate(payload['cache_root'])
        for records in payload['splits'].values():
            for r in records:r['state_path']=relocate(r['state_path'])
        package=load_lazy_package(index,payload=payload)
        for name in ['validation_normal','test_normal','test_positive']:
            split=package['splits'][name]
            missing=[str(split._cache_path(i)) for i in range(len(split)) if not split._cache_path(i).exists()]
            if missing:raise FileNotFoundError('Existing graph caches required; no dataset writes: '+missing[0])
        return package
    geometry_module.load_graph_dataset=relocated_load
    source=root/'results/ego_immune_riskbench_standalone_20260906'
    saved=pd.read_csv(source/'events.csv',float_precision='round_trip')
    saved=saved[(saved.budget==.1)&(saved.regime=='all_safe_recalibrated')&saved.method.isin(['rir_all_safe_selected','cpa_peak'])]
    selected=pd.read_csv(source/'selected_parameters.csv',float_precision='round_trip');selected=selected[selected.budget==.1]
    th=pd.read_csv(source/'thresholds.csv',float_precision='round_trip');th=th[th.budget==.1]
    rows=[];frame_rows=[];max_error=0.
    for fold in range(5):
        index=d.DEFAULT_INDEX_ROOT/f'fold_{fold:02d}/riskbench_nested_fold_index.json'
        observations=d._observations_by_frame(index,splits=('validation_normal','test_normal','test_positive'))
        geometry=d._geometry_from_observations(observations,horizon_seconds=2.,clearance_m=3.5)
        for seed in d.DEFAULT_SEEDS:
            cache=d.DEFAULT_SOURCE_ROOT/f'seed_{seed}/pair_frame_cache/graph_free_one_class/fold_{fold:02d}.pkl'
            frames=d._load_frames(d.DEFAULT_SOURCE_ROOT,seed,fold)
            events={sp:d._event_frames(frames[sp]) for sp in ['validation_normal','test_normal','test_positive']}
            val=events['validation_normal']
            rec=selected[(selected.seed==seed)&(selected.fold==fold)].iloc[0]
            candidate=a.g.Candidate(**{k:float(rec[k]) if k!='recovery_warmup' else int(rec[k]) for k in ['persistence_decay','weight_stress','weight_damage','weight_persistence','recovery_beta','recovery_warmup']})
            vc=d._merged_components(val,d._recompute_components(val,persistence_decay=candidate.persistence_decay,recovery_warmup=candidate.recovery_warmup))
            calibration=d._calibration(val,vc)
            cpa_ref=d._cpa_reference(val,geometry)
            raw_ref=np.sort([v['causal_raw'] for fs in val.values() for f in fs for v in f.get('relation_components',{}).values() if int(v.get('ego_outgoing',0))])
            expected=saved[(saved.seed==seed)&(saved.fold==fold)].set_index(['method','event_uid'])
            for sp in ['test_normal','test_positive']:
                ev=events[sp]
                scores={'rir_all_safe_selected':a.head_scores(ev,geometry,candidate,vc,calibration,cpa_ref,raw_ref),
                        'cpa_peak':a.simple_controls(ev,geometry,cpa_ref,raw_ref)['cpa_peak']}
                for method,mapping in scores.items():
                    threshold=float(th[(th.seed==seed)&(th.fold==fold)&(th.method==method)].iloc[0].threshold)
                    for uid,fs in ev.items():
                        old=expected.loc[(method,uid)]
                        assert float(old.threshold)==threshold
                        tr=[(f,float(mapping[(uid,int(f['frame_position']))])) for f in d._usable_frames(fs)]
                        score=max((v for _,v in tr),default=0.)
                        error=abs(score-float(old.score));max_error=max(max_error,error)
                        assert error<1e-10,(seed,fold,method,uid,error)
                        hit=int(score>=threshold);assert hit==int(old.hit)
                        for n in [20,10,5]:
                            h=int(any(v>=threshold and a.lead(f)>=n for f,v in tr))
                            assert h==int(old[f'hit_by_{n}']),(seed,fold,method,uid,n,h,int(old[f'hit_by_{n}']),threshold,[(a.lead(f),v) for f,v in tr if a.lead(f)>=n])
                        alerts=[f for f,v in tr if v>=threshold]
                        first=min(alerts,key=lambda f:int(f['frame_position'])) if alerts else None
                        lead_frames=a.lead(first) if first else np.nan
                        rows.append(dict(method=method,seed=seed,fold=fold,event_uid=uid,family=old.family,positive=int(old.positive),hit=hit,score=score,threshold=threshold,first_alert_lead_frames=lead_frames,first_alert_lead_seconds=lead_frames/20,first_alert_frame=int(first['frame_position']) if first else np.nan))
                        if int(old.positive):
                            frame_rows.extend(dict(method=method,seed=seed,fold=fold,event_uid=uid,lead_frames=a.lead(f),score=v,threshold=threshold) for f,v in tr)
            print(f'Reproduced fold {fold}, seed {seed}: scores, hits, all 3 deadlines.',flush=True)
    recovered=pd.DataFrame(rows)
    ca_path=root/'results/riskbench_rollout_standalone_closeout_v2/events.csv'
    ca=pd.read_csv(ca_path,float_precision='round_trip');ca=ca[(ca.budget==.1)&(ca.method=='rollout_ca_shared_risk')]
    for r in ca.itertuples():
        assert int(bool(r.hit) and r.lead>=.5-1e-9)==r.hit_by_10
        if r.hit:assert abs(r.lead*20-round(r.lead*20))<1e-8
        rows.append(dict(method='rollout_ca_shared_risk',seed=0,fold=r.fold,event_uid=r.event_uid,family=r.family,positive=r.positive,hit=r.hit,score=r.score,threshold=r.threshold,first_alert_lead_frames=round(r.lead*20) if r.hit else np.nan,first_alert_lead_seconds=r.lead if r.hit else np.nan,first_alert_frame=np.nan))
    combined=pd.DataFrame(rows)
    for method,data in combined[combined.positive==1].groupby('method'):
        assert data.groupby('seed').family.nunique().eq(147).all(),method
    # Pointwise 20-Hz deadlines; no interpolation or threshold sweep.
    curves=[]
    for (method,seed),data in combined[combined.positive==1].groupby(['method','seed']):
        for lead_frames in range(30,0,-1):
            hits=((data.hit==1)&(data.first_alert_lead_frames>=lead_frames)).sum()
            curves.append(dict(method=method,seed=seed,lead_frames=lead_frames,lead_seconds=lead_frames/20,detected=int(hits),danger_denominator=147,recall_percent=float(hits/147*100)))
    curve=pd.DataFrame(curves)
    for method,data in curve.groupby('method'):
        rates=data.groupby('lead_frames').recall_percent.mean()
        assert (np.diff(rates.values)<=1e-10).all()
        full=combined[(combined.method==method)&(combined.positive==1)].groupby('seed').hit.mean().mean()*100
        assert abs(rates.loc[1]-full)<1e-9,(method,full,rates.loc[1])
    out.mkdir(parents=True,exist_ok=True)
    combined.to_csv(out/'first_alert_events.csv',index=False)
    pd.DataFrame(frame_rows).to_csv(out/'replayed_positive_frame_scores.csv.gz',index=False,compression='gzip')
    curve.to_csv(out/'early_recall_curves_per_seed.csv',index=False)
    sources=[source/'selected_parameters.csv',source/'thresholds.csv',source/'events.csv',ca_path]
    manifest=dict(protocol='Existing standalone 10% all-safe selection; original 30 pre-anchor frames; all 147 dangers',training=False,parameter_selection=False,threshold_refitting=False,interpolation=False,source_event_peaks_and_binary_indicators_reproduced=True,max_event_score_absolute_error=max_error,source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'verification.json').write_text(json.dumps(manifest,indent=2))
    print('Verified dense curves. Maximum original score error:',max_error,flush=True)

if __name__=='__main__':main()
