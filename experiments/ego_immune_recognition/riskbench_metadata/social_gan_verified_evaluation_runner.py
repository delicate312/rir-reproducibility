"""Evaluate only after all 15 checkpoints and the convergence audit are frozen."""
import json,hashlib,sys,subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd,torch
from . import social_gan_verified_evaluation as e
R=e.PROJECT_ROOT/'results';root=R/'riskbench_social_gan_verified_v2';O=R/'riskbench_social_gan_verified_evaluation_v2';O.mkdir(exist_ok=True)
assert (root/'training_completion.json').exists() and (root/'convergence_check_complete.json').exists()
if len(sys.argv)>1:
 training_seed,fold=map(int,sys.argv[1:]);torch.set_num_threads(1)
 a,b,c=e.score_fold(fold=fold,index_root=e.DEFAULT_INDEX_ROOT,pair_damp_root=e.DEFAULT_PAIR_DAMP_ROOT,checkpoint_root=root/f'seed_{training_seed}',samples=20,device=torch.device('cpu'))
 (O/f'seed_{training_seed}_fold_{fold}.json').write_text(json.dumps({'events':a,'thresholds':b,'localization':c}))
else:
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('seed_*/fold_*/generator_safe_selection_best.pt')};assert len(hashes)==15
 freeze={'checkpoints':hashes,'samples':20,'risk_scoring':'unchanged shared CV/CA soft center encounter','origin':'fixed zero','calibration':'same reference-negative hard-safe validation','seed_mapping':'training 20260913/14/15 aligned to reference 20260812/13/14 by replicate index'}
 path=O/'evaluation_freeze.json'
 if path.exists():assert json.loads(path.read_text())==freeze,'Changed frozen inputs: choose a fresh output directory'
 else:path.write_text(json.dumps(freeze,indent=2))
 def work(job):
  training_seed,fold=job;saved=O/f'seed_{training_seed}_fold_{fold}.json'
  if not saved.exists():
   with (O/f'run_{training_seed}_{fold}.log').open('w') as log:subprocess.run([sys.executable,'-m',__package__+'.social_gan_verified_evaluation_runner',str(training_seed),str(fold)],stdout=log,stderr=subprocess.STDOUT,check=True)
  print('EVALUATED',training_seed,fold,flush=True)
  return training_seed,json.loads(saved.read_text())
 with ThreadPoolExecutor(max_workers=3) as pool:payloads=list(pool.map(work,[(seed,fold) for seed in [20260913,20260914,20260915] for fold in range(5)]))
 rows=[];thresholds=[];loc=[]
 for training_seed,payload in payloads:
  ref_seed=e.DEFAULT_SEEDS[training_seed-20260913]
  rows.extend([{**x,'seed':ref_seed,'training_seed':training_seed} for x in payload['events']]);thresholds.extend([{**x,'seed':ref_seed,'training_seed':training_seed} for x in payload['thresholds']]);loc.extend([{**x,'seed':ref_seed,'training_seed':training_seed} for x in payload['localization']])
 
 for frame,name in [(pd.DataFrame(rows),'all_seed_events'),(pd.DataFrame(thresholds),'all_seed_thresholds'),(pd.DataFrame(loc),'localization_events')]:
  if name=='all_seed_thresholds':
   # The inherited field names refer to complementary calibration. Keep them
   # for compatibility and add population-neutral aliases for standalone use.
   frame['validation_population_count']=frame['validation_ttc_negative_count']
   frame['validation_false_alarm_rate']=frame['validation_incremental_fpr']
  for regime in ['complementary','standalone']:
   frame[frame.regime==regime].to_csv(O/(name+('_standalone' if regime=='standalone' else '')+'.csv'),index=False)
 (O/'completion.json').write_text(json.dumps({'status':'complete','training_seeds':3,'folds':5,'event_rows':len(rows)}))
