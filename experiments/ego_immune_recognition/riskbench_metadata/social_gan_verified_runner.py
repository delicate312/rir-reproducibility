from pathlib import Path
import argparse,json,sys,subprocess,os,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from . import social_gan_verified_retrain as r
O=r.PROJECT_ROOT/'results/riskbench_social_gan_verified_v2';O.mkdir(exist_ok=True)
config=dict(seeds=[20260913,20260914,20260915],folds=list(range(5)),epochs=60,patience=10,best_k=10,validation_samples=10,batch_size=8,coordinate_policy='fixed origin; actor predictions already ego-relative',checkpoint_selection='fixed-draw minADE10 on disjoint safe-core families, excludes synthetic origin actor',outer_labels_used_for_configuration=False)
if len(sys.argv)>1:
 seed,fold=map(int,sys.argv[1:]);torch.set_num_threads(1)
 lock=O/f'lock_{seed}_{fold}'
 try:lock.mkdir()
 except FileExistsError:
  while lock.exists():time.sleep(1)
  raise SystemExit(0)
 args=argparse.Namespace(index_root=r.DEFAULT_INDEX_ROOT,output_root=O/f'seed_{seed}',fold=fold,seed=seed,epochs=int(os.environ.get("SGAN_EPOCH_CAP", "60")),patience=10,batch_size=8,best_k=10,validation_samples=10,device='cpu',smoke=False)
 try:result=r.run(args)
 finally:lock.rmdir()
 print('FINISHED',seed,fold,'epochs',len(result['history']),'best',result['best_epoch'],'ADE',result['selection_min_ade_m'],flush=True)
else:
 if not (O/'training_freeze.json').exists():(O/'training_freeze.json').write_text(json.dumps(config,indent=2))
 def work(job):
  seed,fold=job;target=O/f'seed_{seed}'/f'fold_{fold:02d}'/'training_summary.json'
  if not target.exists():
   with (O/f'train_{seed}_{fold}.log').open('w') as log:subprocess.run([sys.executable,'-m',__package__+'.social_gan_verified_runner',str(seed),str(fold)],stdout=log,stderr=subprocess.STDOUT,check=True)
  x=json.loads(target.read_text());print('FINISHED',seed,fold,'epochs',len(x['history']),'best',x['best_epoch'],'ADE',x['selection_min_ade_m'],flush=True);return x
 with ThreadPoolExecutor(max_workers=3) as pool:summaries=list(pool.map(work,[(seed,fold) for seed in config['seeds'] for fold in config['folds']]))
 (O/'training_completion.json').write_text(json.dumps({'status':'complete','fits':[{k:v for k,v in x.items() if k!='history'} for x in summaries]},indent=2))
