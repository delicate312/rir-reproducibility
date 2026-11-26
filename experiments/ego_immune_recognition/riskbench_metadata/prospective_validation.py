"""Prospective freeze, integrity checks and label-blind data admission.

A local receipt is not external preregistration. Existing-data replay is only
an engineering check. Admission checks identity/provenance, not label validity.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = ROOT / 'results/rir_prospective_freeze_20260908_v2'
SOURCE = ROOT / 'results/ego_immune_riskbench_standalone_20260906'
SEEDS = (20260812, 20260813, 20260814)
METHODS = ('rir_all_safe_selected', 'cpa_peak', 'cpa_temporal', 'cpa_ema_warmup',
           'cpa_cusum_warmup', 'constant_damp_control', 'raw_residual_hybrid')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def family_fold(family):
    return int.from_bytes(hashlib.sha256(('rir-prospective-v1|'+family).encode()).digest()[:8], 'big') % 5


def verify(bundle, expected_sha256=None):
    bundle = Path(bundle)
    manifest = bundle / 'manifest.json'
    recorded = (bundle / 'MANIFEST.sha256').read_text().strip()
    if expected_sha256 is not None and recorded != expected_sha256:
        raise ValueError('Bundle differs from independently recorded manifest digest')
    if digest(manifest) != recorded:
        raise ValueError('Manifest digest mismatch')
    payload = json.loads(manifest.read_text())
    for name, expected in payload['files'].items():
        path = (bundle/name).resolve()
        if not path.is_relative_to(bundle.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError(f'Frozen artifact missing or changed: {name}')
    return payload


def inventory():
    """Conservative exclusion: all available old raw families, not just test IDs."""
    raw = ROOT/'data/riskbench/raw/metadata/state'
    rows=[]
    for path in sorted(raw.glob('*/*.json')):
        base=path.stem.rstrip('_').rsplit('_',2)[0]
        rows.append(dict(category=path.parent.name, scenario_id=path.stem,
                         family=f'{path.parent.name}:{base}', base_scenario_id=base,
                         state_sha256=digest(path)))
    historical=[]
    for path in sorted((ROOT/'data/processed').rglob('*index*.json')):
        obj=json.loads(path.read_text())
        if not isinstance(obj,dict) or not isinstance(obj.get('splits'),dict):
            continue
        for split, entries in obj['splits'].items():
            if not isinstance(entries,list):
                continue
            for r in entries:
                if isinstance(r,dict) and r.get('scenario_group_id'):
                    historical.append(dict(index=str(path.relative_to(ROOT)),split=split,
                                           family=r['scenario_group_id'],base_scenario_id=r.get('base_scenario_id','')))
    return dict(raw_records=rows,historical_index_records=historical,
                policy='Exclude every currently available raw family conservatively; availability is not proof every file was scored.')


def frozen_scores(bundle, seed, fold, budget, events, geometry):
    """Score cached causal forecaster outputs using exported references, no fitting."""
    import numpy as np
    from . import damp_decision_necessity_ablation as d
    from .standalone_rir_audit import simple_controls
    from .history_geometry_validation import pair_and_scene_scores
    base=Path(bundle)/'calibration'/f'{seed}_{fold}_{int(budget*100)}'
    cfg=json.loads(base.with_suffix('.json').read_text())
    refs=np.load(base.with_suffix('.npz'),allow_pickle=False)
    candidate=d.Candidate(**cfg['candidate'])
    components=d._merged_components(events,d._recompute_components(events,
        persistence_decay=candidate.persistence_decay,recovery_warmup=candidate.recovery_warmup))
    activation=d._activation_fn('rir_nested_selected',cfg['component_calibration'],candidate)
    _,scene=pair_and_scene_scores(events,components,geometry,refs['cpa'],activation,refs['damp'],activation,refs['damp'],refs['raw'])
    scores=simple_controls(events,geometry,refs['cpa'],refs['raw'])
    scores['rir_all_safe_selected']={k:v['rir'] for k,v in scene.items()}
    return scores,cfg['thresholds']


def freeze(bundle):
    # Numerical references are reconstructed from OLD validation-normal only.
    import numpy as np
    from dataclasses import asdict
    from . import damp_decision_necessity_ablation as d
    from .standalone_rir_audit import OUT
    bundle=Path(bundle)
    if bundle.exists():
        raise ValueError('Refuse to overwrite a freeze bundle; verify the existing one')
    work=bundle.with_name(bundle.name+'.building')
    work.mkdir()  # Refuse to silently reuse a failed build.
    (work/'calibration').mkdir();(work/'checkpoints').mkdir();(work/'source_results').mkdir()
    inv=inventory();write_json(work/'prior_data_inventory.json',inv)
    selected=list(csv.DictReader((OUT/'selected_parameters.csv').open()))
    thresholds=list(csv.DictReader((OUT/'thresholds.csv').open()))
    for name in ['selected_parameters.csv','thresholds.csv','protocol.json','completion.json','analysis_completion.json','events.csv','seed_metrics.csv']:
        shutil.copy2(SOURCE/name,work/'source_results'/name)
    checkpoint_rows=[];provenance=[]
    for fold in range(5):
        index=d.DEFAULT_INDEX_ROOT/f'fold_{fold:02d}/riskbench_nested_fold_index.json'
        observations=d._observations_by_frame(index,splits=('validation_normal',))
        geometry=d._geometry_from_observations(observations,horizon_seconds=2.,clearance_m=3.5)
        for seed in SEEDS:
            receipt=d.DEFAULT_SOURCE_ROOT/f'seed_{seed}/pre_outer_test_checkpoint_freeze.json'
            row=next(r for r in json.loads(receipt.read_text())['checkpoints'] if r['fold']==fold)
            cp=Path(row['checkpoint_path'])
            if digest(cp)!=row['checkpoint_sha256']:
                raise ValueError(f'Historical checkpoint changed: {cp}')
            shutil.copy2(cp,work/'checkpoints'/f'{seed}_{fold}.pt')
            checkpoint_rows.append(row)
            frames=d._load_frames(d.DEFAULT_SOURCE_ROOT,seed,fold)
            val=d._event_frames(frames['validation_normal'])
            cpa=d._cpa_reference(val,geometry)
            raw=np.sort([v['causal_raw'] for fs in val.values() for f in fs for v in f.get('relation_components',{}).values() if int(v.get('ego_outgoing',0))])
            cache=d.DEFAULT_SOURCE_ROOT/f'seed_{seed}/pair_frame_cache/graph_free_one_class/fold_{fold:02d}.pkl'
            provenance.append(dict(seed=seed,fold=fold,index=str(index),index_sha256=digest(index),
                cache=str(cache),cache_sha256=digest(cache),validation_event_uids=sorted(val)))
            for budget in (.05,.10):
                chosen=next(r for r in selected if int(r['seed'])==seed and int(r['fold'])==fold and float(r['budget'])==budget)
                candidate=d.Candidate(**{k:(int(chosen[k]) if k=='recovery_warmup' else float(chosen[k])) for k in asdict(d.SIMPLE)})
                vc=d._merged_components(val,d._recompute_components(val,persistence_decay=candidate.persistence_decay,recovery_warmup=candidate.recovery_warmup))
                calibration=d._calibration(val,vc)
                activation=d._activation_fn('rir_nested_selected',calibration,candidate)
                damp=d._activation_reference(vc,activation)
                th={r['method']:float(r['threshold']) for r in thresholds if int(r['seed'])==seed and int(r['fold'])==fold and float(r['budget'])==budget and r['method'] in METHODS}
                assert set(th)==set(METHODS)
                base=work/'calibration'/f'{seed}_{fold}_{int(budget*100)}'
                np.savez_compressed(base.with_suffix('.npz'),cpa=cpa,raw=raw,damp=damp)
                write_json(base.with_suffix('.json'),dict(candidate=asdict(candidate),component_calibration=calibration,thresholds=th,seed=seed,fold=fold,budget=budget))
            print(f'Frozen seed {seed}, fold {fold}: checkpoint + old-safe reference distributions + thresholds',flush=True)
    write_json(work/'provenance.json',dict(checkpoints=checkpoint_rows,calibration=provenance))
    # Copy source dependencies, not only their mutable working-tree locations.
    for root in ('experiments','models','utils'):
        for path in sorted((ROOT/root).rglob('*.py')):
            dest=work/'code'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    for path in ROOT.glob('*.py'):
        dest=work/'code'/path.name;shutil.copy2(path,dest)
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','torch','torch-geometric']}
    write_json(work/'environment.json',dict(python=sys.version,executable=sys.executable,packages=versions))
    plan=dict(version='prospective-validation-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        status='Frozen before NEW data; no independent performance result exists yet',
        model='Existing graph-free RIR; inverse-TTC inputs and CPA alignment retained',
        scope='Standalone primary; complementary original results retained as development evidence',
        primary_budget=.10,secondary_budget=.05,methods=list(METHODS),
        primary_contrasts=['rir_all_safe_selected - cpa_peak','rir_all_safe_selected - raw_residual_hybrid'],
        primary_metrics=['event recall','all-hard-safe event false-alarm rate'],
        inference='10,000 paired family bootstrap; average seed binary outcomes per family first; two-sided paired sign-flip (exact <=18 nonzero families, else 10,000 plus-one); Holm across 4 primary tests (2 comparators x 2 metrics)',
        secondary='5% source validation budget; recall >=0.5s pre-anchor; other CPA/content controls, AP; descriptive pointwise intervals, no confirmatory significance claim',
        statistical_seed=20260908,
        success_interpretation='Claim higher recall only for corrected primary contrast; report FAR and CI jointly. No matched target-budget claim if costs differ. No conjunctive claim that both recall and FAR improve unless both supported.',
        routing='sha256(UTF8("rir-prospective-v1|" + canonical family id)), first 8 bytes big-endian modulo 5; same fold for all 3 seeds. No ensemble-score threshold.',
        sample_plan=dict(target_danger_families=150,target_hard_safe_families=300,unit='one prespecified representative event per independent family',
            stopping='Fixed family counts, not significance. Any smaller resource-limited collection declared before model scoring; report as pilot.',
            rationale='Feasibility/precision target, not a formal power guarantee; worst-case 95% binomial half-width roughly 8 and 6 percentage points.'),
        adaptation='None in primary analysis. No new means, CDFs, coefficients, thresholds, epoch selection, or model retraining.',
        eligibility='New canonical parent scripts/routes/interactions, not weather/density/camera variants, renamed old families, duplicated trajectories, or families previously used for design.',
        collection='Sample/label/QC without model scores; retain low-visibility and missed/empty-window events; never relabel hard cases based on RIR or CPA.',
        time_contract='20Hz causal ego-relative x/y in metres, stable actor IDs; current/past finite differences; 40 pre-anchor input frames; same final 30-frame detection window',
        labels='Collision anchor from independent ground truth; hard-safe interactive episodes with prespecified interaction anchor and no collision during a complete declared observation horizon. No CPA/RIR threshold used as label.',
        publication='Local hash receipt only. Record manifest digest with supervisor or public preregistration BEFORE collecting/opening new evaluation data; no automatic publication performed.',
        readiness='Calibration/export/replay and admission tooling prepared. New-data adapter and raw-to-model integration must be verified before any independent scoring.')
    write_json(work/'protocol.json',plan)
    write_json(work/'manifest.json',dict(created_utc=plan['created_utc'],files={str(p.relative_to(work)):digest(p) for p in sorted(work.rglob('*')) if p.is_file()}))
    (work/'MANIFEST.sha256').write_text(digest(work/'manifest.json')+'\n')
    work.rename(bundle)
    verify(bundle)
    print('Bundle sealed:',bundle,flush=True)


def admit(bundle, candidate, output):
    """Reject known/duplicate families and incomplete collection attestations.

    A passing receipt authorizes adapter engineering only, not proof of novelty.
    Raw data and labels remain to be validated by a collector/model-blind reviewer.
    """
    verify(bundle)
    inv=json.loads((Path(bundle)/'prior_data_inventory.json').read_text())
    obj=json.loads(Path(candidate).read_text())
    frozen=json.loads((Path(bundle)/'protocol.json').read_text())['created_utc']
    created=datetime.fromisoformat(obj['collection_started_utc'])
    if created.tzinfo is None or created<=datetime.fromisoformat(frozen):
        raise ValueError('Collection must start after the prospective freeze, with timezone')
    for key in ('model_scores_not_used','labels_sealed_before_scoring','new_parent_families','identity_review_completed'):
        if obj.get(key) is not True:
            raise ValueError(f'Required collector attestation absent: {key}')
    if not obj.get('collector') or not obj.get('provenance_description'):
        raise ValueError('Collector identity and provenance description required')
    records=obj['records']
    if not records:
        raise ValueError('Empty candidate manifest')
    blocked={r['family'] for r in inv['raw_records']}|{r['family'] for r in inv['historical_index_records']}
    bases={r['base_scenario_id'] for r in inv['raw_records']}|{r['base_scenario_id'] for r in inv['historical_index_records'] if r['base_scenario_id']}
    hashes={r['state_sha256'] for r in inv['raw_records']}
    seen_families=set();seen_hashes=set();accepted=[]
    for r in records:
        family=r['canonical_family_id'];parent=r['parent_script_id']
        if not family or not parent or family in blocked or parent in bases or family in seen_families:
            raise ValueError(f'Known, duplicate or missing parent family: {family}')
        path=Path(r['state_path'])
        if not path.is_absolute():
            path=Path(candidate).resolve().parent/path
        actual=digest(path)
        if actual!=r['state_sha256'] or actual in hashes or actual in seen_hashes:
            raise ValueError(f'Changed or reused raw state: {family}')
        seen_families.add(family);seen_hashes.add(actual)
        accepted.append(dict(canonical_family_id=family,parent_script_id=parent,state_path=str(path.resolve()),state_sha256=actual,fold=family_fold(family)))
    out=Path(output)
    if out.exists():
        raise ValueError('Do not overwrite an admission receipt')
    write_json(out,dict(status='identity_checks_passed_adapter_and_label_QC_still_required',
        created_utc=datetime.now(timezone.utc).isoformat(),candidate_sha256=digest(candidate),
        freeze_manifest_sha256=digest(Path(bundle)/'manifest.json'),records=accepted))
    return len(accepted)


def replay(bundle, output):
    """OLD data only: exported scores must reproduce saved development results."""
    verify(bundle)
    # Refuse to execute changed working-tree dependencies against a frozen receipt.
    manifest = verify(bundle)
    for name, expected_hash in manifest['files'].items():
        if name.startswith('code/') and digest(ROOT/name[5:]) != expected_hash:
            raise ValueError(f'Runtime source differs from freeze: {name}')
    import numpy as np
    from . import damp_decision_necessity_ablation as d
    expected=list(csv.DictReader((Path(bundle)/'source_results/events.csv').open()))
    expected={(int(r['seed']),int(r['fold']),float(r['budget']),r['method'],r['event_uid']):r for r in expected if r['regime']=='all_safe_recalibrated' and r['method'] in METHODS}
    checked=0;max_error=0.
    for fold in range(5):
        index=d.DEFAULT_INDEX_ROOT/f'fold_{fold:02d}/riskbench_nested_fold_index.json'
        geometry=d._geometry_from_observations(d._observations_by_frame(index,splits=('test_normal','test_positive')),horizon_seconds=2.,clearance_m=3.5)
        for seed in SEEDS:
            fs=d._load_frames(d.DEFAULT_SOURCE_ROOT,seed,fold)
            events={**d._event_frames(fs['test_normal']),**d._event_frames(fs['test_positive'])}
            for budget in (.05,.10):
                scores,thresholds=frozen_scores(bundle,seed,fold,budget,events,geometry)
                for name in METHODS:
                    for uid,frames in events.items():
                        value=d.event_score([(f,scores[name][(uid,int(f['frame_position']))]) for f in d._usable_frames(frames)])
                        row=expected[seed,fold,budget,name,uid]
                        err=abs(value-float(row['score']));max_error=max(max_error,err)
                        np.testing.assert_allclose(value,float(row['score']),atol=1e-10,rtol=0)
                        assert thresholds[name]==float(row['threshold'])
                        assert int(value>=thresholds[name])==int(row['hit'])
                        checked+=1
            print(f'Replayed OLD fold {fold}, seed {seed}',flush=True)
    out=Path(output)
    if out.exists():raise ValueError('Replay report already exists')
    write_json(out,dict(status='passed',purpose='Old-data engineering reproduction ONLY; NOT independent validation',event_method_budget_seed_checks=checked,max_score_error=max_error,freeze_manifest_sha256=digest(Path(bundle)/'manifest.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze','verify','admit','replay'])
    parser.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--expected-sha256')
    args=parser.parse_args()
    if args.command=='freeze':freeze(args.bundle)
    elif args.command=='verify':print(json.dumps(dict(status='verified',file_count=len(verify(args.bundle,args.expected_sha256)['files']),manifest_sha256=digest(args.bundle/'manifest.json'))))
    elif args.command=='admit':
        if args.candidate is None or args.output is None:parser.error('--candidate and --output required')
        print('Identity-checked records:',admit(args.bundle,args.candidate,args.output))
    else:
        if args.output is None:parser.error('--output required')
        replay(args.bundle,args.output)

if __name__=='__main__':main()
