"""Standalone RIR: remove TTC union and calibrate on ALL hard-safe events.

Three distinct interventions: frozen original head/threshold with no union;
original head recalibrated on all hard-safe; fresh all-hard-safe head selection
using the existing 90-candidate grid. No positive labels enter selection.
The underlying learned relation features still contain inverse TTC, and CPA
alignment/bypass remain. This is not an inverse-TTC feature-removal experiment.
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import damp_decision_necessity_ablation as d
from . import damp_hard_safe_grid_selection as g
from .damp_incremental_value_audit import BUDGETS
from .history_geometry_validation import pair_and_scene_scores, history_controls, lead

OUT = d.PROJECT_ROOT / 'results/ego_immune_riskbench_standalone_20260906'
HISTORY_OUT = d.PROJECT_ROOT / 'results/ego_immune_riskbench_history_geometry_20260906'


def vector_layout(events, components, geometry):
    """Flatten calibration components and causal three-observation indices."""
    uids = sorted(events)
    lookup, raw, component_events = {}, [], []
    for ei, uid in enumerate(uids):
        for pos, targets in components[uid].items():
            for target, values in targets.items():
                lookup[(uid, pos, target)] = len(raw)
                raw.append([values[k] for k in ('stress_raw','damage_raw','persistence_raw','recovery_raw')])
                component_events.append(ei)
    geo_raw, geo_events, pair_raw, pair_components, pair_events, history_indices, valid = [], [], [], [], [], [], []
    for ei, uid in enumerate(uids):
        for frame in events[uid]:
            pos = int(frame['frame_position'])
            for item in geometry.get((uid,pos), {}).values():
                geo_raw.append(item['raw'])
                geo_events.append(ei)
        histories = {}
        for frame in d._usable_frames(events[uid]):
            pos = int(frame['frame_position'])
            for target, item in geometry.get((uid,pos), {}).items():
                key = (uid,pos,str(target))
                if key not in lookup:
                    continue
                h = histories.setdefault(str(target), [])
                if h and pos != h[-1][0]+1:
                    h.clear()
                i = len(pair_raw)
                pair_raw.append(item['raw'])
                pair_components.append(lookup[key])
                pair_events.append(ei)
                h.append((pos,i))
                h[:] = h[-3:]
                valid.append(len(h)==3)
                history_indices.append([k for _,k in h] if len(h)==3 else [i,i,i])
    return dict(uids=uids, raw=np.asarray(raw,float), component_events=np.asarray(component_events,int),
        geo_raw=np.asarray(geo_raw,float),geo_events=np.asarray(geo_events,int),pair_raw=np.asarray(pair_raw,float),
        pair_components=np.asarray(pair_components,int),pair_events=np.asarray(pair_events,int),
        history_indices=np.asarray(history_indices,int).reshape(-1,3),valid=np.asarray(valid,bool))


def vector_scores(layout, train_indices, candidate):
    """All event scores under references fitted to selected safe events only."""
    raw = layout['raw']
    train_mask = np.isin(layout['component_events'], train_indices)
    reference = raw[train_mask]
    centers = np.median(reference,axis=0)
    scales = np.maximum(np.maximum(1.4826*np.median(np.abs(reference-centers),axis=0),np.quantile(reference,.9,axis=0)-centers),1e-3)
    z = np.maximum((raw-centers)/scales,0.)
    activation = np.maximum(candidate.weight_stress*z[:,0]+candidate.weight_damage*z[:,1]+candidate.weight_persistence*z[:,2]-candidate.recovery_beta*z[:,3],0.)
    damp_ref = np.sort(activation[train_mask])
    dp = np.searchsorted(damp_ref,activation[layout['pair_components']],side='right')/len(damp_ref)
    cpa_ref = np.sort(layout['geo_raw'][np.isin(layout['geo_events'],train_indices)])
    cp = np.searchsorted(cpa_ref,layout['pair_raw'],side='right')/len(cpa_ref)
    fused = np.sqrt(cp*dp)
    scores = np.where(layout['valid'],fused[layout['history_indices']].mean(axis=1),0.)
    scores = np.where(cp >= d.SEVERE_CPA_PERCENTILE,np.maximum(scores,cp),scores)
    result = np.zeros(len(layout['uids']))
    np.maximum.at(result,layout['pair_events'],scores)
    return result


def select_all_safe(events, geometry, index, seed, fold):
    subfolds = g._balanced_subfolds(events,g._event_family_map(index),seed=seed,fold=fold,count=5)
    candidates = g.candidate_grid()
    layouts = {}
    rows = []
    for candidate in candidates:
        key = (candidate.persistence_decay,candidate.recovery_warmup)
        if key not in layouts:
            comp = d._recompute_components(events,persistence_decay=key[0],recovery_warmup=key[1])
            layouts[key] = vector_layout(events,comp,geometry)
        layout = layouts[key]
        uids = layout['uids']
        splits = []
        for heldout_uids in subfolds:
            heldout = np.array([i for i,uid in enumerate(uids) if uid in heldout_uids])
            train = np.array([i for i,uid in enumerate(uids) if uid not in heldout_uids])
            scores = vector_scores(layout,train,candidate)
            splits.append((scores[train],scores[heldout]))
        for budget in BUDGETS:
            rates, counts, totals = [], [], []
            for training, heldout in splits:
                threshold, achieved = d.threshold_at_event_fpr(training,budget)
                assert achieved <= budget+1e-12
                count = int(np.count_nonzero(heldout>=threshold))
                counts.append(count)
                totals.append(len(heldout))
                rates.append(count/len(heldout))
            pooled = sum(counts)/sum(totals)
            rows.append(dict(seed=seed,fold=fold,budget=budget,**asdict(candidate),candidate_id=candidate.candidate_id,
                calibration_error=abs(pooled-budget),heldout_fpr_std=np.std(rates,ddof=1),heldout_fpr_pooled=pooled,
                within_budget=int(pooled<=budget+1e-12),uniform_weight_distance=candidate.uniform_distance,
                subfold_rates=json.dumps(rates),subfold_n=json.dumps(totals)))
    selected = {}
    selected_rows = []
    for budget in BUDGETS:
        eligible = [r for r in rows if r['budget']==budget and r['within_budget']]
        pool = [r for r in rows if r['budget']==budget]
        chosen = min(eligible or pool,key=g._selection_key)
        selected[budget] = g.Candidate(**{k:chosen[k] for k in asdict(candidates[0])})
        selected_rows.append(dict(chosen,fallback=int(not eligible)))
    return selected,rows,selected_rows


def head_scores(events, geometry, candidate, validation_components, calibration, cpa_ref, raw_ref):
    components = d._merged_components(events,d._recompute_components(events,persistence_decay=candidate.persistence_decay,recovery_warmup=candidate.recovery_warmup))
    activation = d._activation_fn('rir_nested_selected',calibration,candidate)
    nr = d._activation_fn('rir_nested_selected',calibration,replace(candidate,recovery_beta=0.))
    damp_ref = d._activation_reference(validation_components,activation)
    nr_ref = d._activation_reference(validation_components,nr)
    _,scene = pair_and_scene_scores(events,components,geometry,cpa_ref,activation,damp_ref,nr,nr_ref,raw_ref)
    return {key:vals['rir'] for key,vals in scene.items()}


def simple_controls(events, geometry, cpa_ref, raw_ref):
    names=('cpa_peak','cpa_temporal','cpa_ema_warmup','cpa_cusum_warmup','constant_damp_control','raw_residual_hybrid')
    scene={}
    for uid,fs in events.items():
        actors={}
        frame_values={}
        for f in sorted(fs,key=lambda f:f['frame_position']):
            if not -20<=lead(f)<=40:
                continue
            pos=int(f['frame_position'])
            scene[(uid,pos)]=dict.fromkeys(names,0.)
            frame_values[pos]={str(v['target_id']):v for v in f.get('relation_components',{}).values() if int(v.get('ego_outgoing',0))}
            for actor,geo in geometry.get((uid,pos),{}).items():
                actors.setdefault(str(actor),[]).append((pos,lead(f),d._percentile(cpa_ref,geo['raw'])))
        for actor,rows in actors.items():
            ctrl=history_controls([p for p,_,_ in rows],[c for _,_,c in rows])
            short_rows=[(p,c) for p,l,c in rows if l<=30]
            short={p:control for (p,_),control in zip(short_rows,history_controls([p for p,_ in short_rows],[c for _,c in short_rows]))}
            h=[]
            for (pos,l,cp),history in zip(rows,ctrl):
                if l>30:
                    continue
                current=dict(cpa_peak=cp,cpa_temporal=short[pos][1],cpa_ema_warmup=history[2],cpa_cusum_warmup=history[4],constant_damp_control=0.,raw_residual_hybrid=0.)
                values=frame_values[pos].get(actor)
                if values is None:
                    h=[]
                else:
                    if h and pos!=h[-1][0]+1:
                        h=[]
                    raw=d._percentile(raw_ref,values['causal_raw'])
                    h.append((pos,np.sqrt(cp),np.sqrt(cp*raw)))
                    h=h[-3:]
                    score=np.mean(np.array(h)[:,1:],axis=0) if len(h)==3 else np.zeros(2)
                    if cp>=d.SEVERE_CPA_PERCENTILE:
                        score=np.maximum(score,cp)
                    current['constant_damp_control'],current['raw_residual_hybrid']=score
                for name in names:
                    scene[(uid,pos)][name]=max(scene[(uid,pos)][name],current[name])
    return {name:{key:values[name] for key,values in scene.items()} for name in names}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=d.sha256_file(Path(__file__)),
        scope='No TTC alarm union or TTC filtering in new calibration/selection. Frozen feature extractor still uses inverse TTC, CPA remains.',
        regimes=['remove_union_only','all_safe_recalibrated'],budgets=list(BUDGETS),seeds=list(d.DEFAULT_SEEDS),
        candidate_grid=[asdict(c) for c in g.candidate_grid()],
        fresh_selector='existing 90-candidate grid; 5 family subfolds of validation_normal ONLY; all events in denominator; deploy-time severe CPA bypass also enabled during selection; calibration-error/stability/simplicity tie-break; preserve fallback if none within inner budget',
        fixed_heads={'rir_fixed_simple':asdict(d.SIMPLE),'rir_fixed_paper':asdict(d.PAPER_CURRENT)},
        original_head='coefficients historically selected on TTC-negative validation; retained solely for faithful ablation, not claimed TTC-independent parameter selection',
        primary='standalone fresh all-safe selected RIR vs peak CPA, temporal CPA, full-warmup EMA/CUSUM, constant DAMP, raw residual; both budgets',
        metrics='recall over all 147 danger families, event false alarms over all 273 hard-safe families, precision within this case-control corpus, AP/AUROC, recall at 1/.5/.25 s; no conditioning on TTC for primary metrics',
        statistics='10000 paired family bootstrap; seed means first for binary metrics; paired randomization p; Holm over all comparisons per metric; internal exploratory evidence',
        frozen_data='same previously examined outer families, no retraining, no outer-positive model/head/threshold selection')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    original=pd.read_csv(HISTORY_OUT/'events.csv')
    event_rows=[];threshold_rows=[];selection_rows=[];grid_rows=[]
    old_references=pd.concat([pd.read_csv(HISTORY_OUT/'system_events.csv'),pd.read_csv(HISTORY_OUT/'mechanism_system_events.csv')]).set_index(['seed','fold','budget','event_uid','method'])
    for fold in range(5):
        index=d.DEFAULT_INDEX_ROOT/f'fold_{fold:02d}/riskbench_nested_fold_index.json'
        observations=d._observations_by_frame(index,splits=('validation_normal','test_normal','test_positive'))
        geometry=d._geometry_from_observations(observations,horizon_seconds=2.,clearance_m=3.5)
        payload=json.loads(index.read_text())
        metadata={f"riskbench:{r['category']}:{r['scenario_id']}":r for sp in ('validation_normal','test_normal','test_positive') for r in payload['splits'][sp]}
        for seed in d.DEFAULT_SEEDS:
            frames=d._load_frames(d.DEFAULT_SOURCE_ROOT,seed,fold)
            events={sp:d._event_frames(frames[sp]) for sp in ('validation_normal','test_normal','test_positive')}
            val=events['validation_normal']
            print(f'fold {fold}, seed {seed}: selecting on ALL {len(val)} hard-safe validation families',flush=True)
            selected,grid,receipts=select_all_safe(val,geometry,index,seed,fold)
            grid_rows.extend(grid);selection_rows.extend(receipts)
            pd.DataFrame(grid_rows).to_csv(OUT/'selection_grid.csv',index=False)
            pd.DataFrame(selection_rows).to_csv(OUT/'selected_parameters.csv',index=False)
            cpa_ref=d._cpa_reference(val,geometry)
            raw_ref=np.sort([v['causal_raw'] for fs in val.values() for f in fs for v in f.get('relation_components',{}).values() if int(v.get('ego_outgoing',0))])
            controls={sp:simple_controls(ev,geometry,cpa_ref,raw_ref) for sp,ev in events.items()}
            for budget,(selection_root,_) in BUDGETS.items():
                old=d._selected_candidate(d.PROJECT_ROOT/'results'/selection_root,seed=seed,fold=fold)
                heads={'rir_original_head':old,'rir_all_safe_selected':selected[budget],'rir_fixed_simple':d.SIMPLE,'rir_fixed_paper':d.PAPER_CURRENT}
                scored={sp:dict(ctrl) for sp,ctrl in controls.items()}
                for name,candidate in heads.items():
                    vc=d._merged_components(val,d._recompute_components(val,persistence_decay=candidate.persistence_decay,recovery_warmup=candidate.recovery_warmup))
                    calibration=d._calibration(val,vc)
                    # Fit all references only to validation_normal, then score.
                    for sp,ev in events.items():
                        scored[sp][name]=head_scores(ev,geometry,candidate,vc,calibration,cpa_ref,raw_ref)
                    if name=='rir_all_safe_selected':
                        layout=vector_layout(val,vc,geometry)
                        fast=vector_scores(layout,np.arange(len(layout['uids'])),candidate)
                        slow=[max((scored['validation_normal'][name][(uid,int(f['frame_position']))] for f in d._usable_frames(val[uid])),default=0.) for uid in layout['uids']]
                        np.testing.assert_allclose(fast,slow,atol=1e-10,rtol=0)
                thresholds={}
                for name,scores in scored['validation_normal'].items():
                    vals=[max((scores[(uid,int(f['frame_position']))] for f in d._usable_frames(fs)),default=0.) for uid,fs in val.items()]
                    th,rate=d.threshold_at_event_fpr(vals,budget)
                    assert rate<=budget+1e-12
                    thresholds[name]=th
                    threshold_rows.append(dict(seed=seed,fold=fold,budget=budget,method=name,threshold=th,val_rate=rate,val_n=len(vals)))
                pd.DataFrame(threshold_rows).to_csv(OUT/'thresholds.csv',index=False)
                for sp in ('test_normal','test_positive'):
                    for uid,fs in events[sp].items():
                        for name,scores in scored[sp].items():
                            tr=[(f,scores[(uid,int(f['frame_position']))]) for f in d._usable_frames(fs)]
                            score=d.event_score(tr)
                            reference_name='rir' if name=='rir_original_head' else name
                            if reference_name in ['rir','cpa_peak','cpa_temporal','cpa_ema_warmup','cpa_cusum_warmup','constant_damp_control','raw_residual_hybrid']:
                                expected=old_references.loc[(seed,fold,budget,uid,reference_name)]
                                np.testing.assert_allclose(score,expected.score,atol=1e-10,rtol=0)
                            for regime,th in [('all_safe_recalibrated',thresholds[name])]+([('remove_union_only',float(old_references.loc[(seed,fold,budget,uid,'rir')].threshold))] if name=='rir_original_head' else []):
                                row=dict(seed=seed,fold=fold,budget=budget,regime=regime,method=name,event_uid=uid,family=metadata[uid]['scenario_group_id'],positive=int(sp=='test_positive'),score=score,threshold=th,hit=int(score>=th))
                                for horizon in (20,10,5):
                                    row[f'hit_by_{horizon}']=int(any(s>=th and lead(f)>=horizon for f,s in tr))
                                event_rows.append(row)
                print(f'fold {fold}, seed {seed}, budget {budget}: standalone scored; 7 original score controls reproduced',flush=True)
        pd.DataFrame(event_rows).to_csv(OUT/'events.csv',index=False)
    (OUT/'completion.json').write_text(json.dumps(dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=d.sha256_file(Path(__file__)),event_rows=len(event_rows),selection_rows=len(selection_rows),grid_rows=len(grid_rows),reproduction='original RIR and six geometry/content controls, both budgets; vector selector scores match direct pipeline'),indent=2))


if __name__=='__main__':
    main()
