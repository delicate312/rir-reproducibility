"""Extend only safe-validation-capped fits; no outer scores are consulted."""
from pathlib import Path
import json,os,sys,subprocess,shutil
from concurrent.futures import ThreadPoolExecutor
from . import social_gan_verified_retrain as r
O=r.PROJECT_ROOT/'results/riskbench_social_gan_verified_v2'
assert (O/'training_completion.json').exists()
config=json.loads((O/'training_freeze.json').read_text());config['safe_only_extension_rule']='If best checkpoint is within 10 epochs of the 60-epoch cap, rerun same seed up to 120 with unchanged patience ten; preserve initial fit. Report any remaining cap-limited fits.';(O/'training_freeze.json').write_text(json.dumps(config,indent=2))
jobs=[]
for f in O.glob('seed_*/fold_*/training_summary.json'):
 x=json.loads(f.read_text())
 if len(x['history'])==60 and x['best_epoch']>50:jobs.append((x['seed'],x['fold'],f.parent))
def work(job):
 seed,fold,path=job;backup=path/'initial_cap60';backup.mkdir(exist_ok=True)
 for name in ['training_summary.json','generator_safe_selection_best.pt']:
  if not (backup/name).exists():shutil.copy2(path/name,backup/name)
 with (O/f'extension_{seed}_{fold}.log').open('w') as log:subprocess.run([sys.executable,'-m',__package__+'.social_gan_verified_runner',str(seed),str(fold)],env=os.environ|{'SGAN_EPOCH_CAP':'120'},stdout=log,stderr=subprocess.STDOUT,check=True)
 x=json.loads((path/'training_summary.json').read_text());print('EXTENDED',seed,fold,len(x['history']),x['best_epoch'],x['selection_min_ade_m'],flush=True)
with ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(work,jobs))
fits=[json.loads(f.read_text()) for f in sorted(O.glob('seed_*/fold_*/training_summary.json'))];assert len(fits)==15
(O/'training_completion.json').write_text(json.dumps({'status':'complete','fits':[{k:v for k,v in x.items() if k!='history'} for x in fits],'safe_only_extended_fits':len(list(O.glob('seed_*/fold_*/initial_cap60'))),'cap_limited_fits':[(x['seed'],x['fold']) for x in fits if len(x['history'])==120 and x['best_epoch']>110]},indent=2))
(O/'convergence_check_complete.json').write_text(json.dumps({'status':'complete','extended_fits':len(list(O.glob('seed_*/fold_*/initial_cap60')))}))
