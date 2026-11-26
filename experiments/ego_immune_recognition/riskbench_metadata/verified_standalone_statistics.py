from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[3]/'results';O=R/'riskbench_verified_strong_baselines_v2';O.mkdir(exist_ok=True)
x=pd.read_csv(R/'ego_immune_riskbench_standalone_20260906/events.csv');x=x[(x.regime=='all_safe_recalibrated')&x.method.isin(['rir_all_safe_selected','cpa_peak'])]
c=pd.read_csv(R/'riskbench_rollout_standalone_closeout_v2/events.csv');c['seed']=0
parts=[x,c]
s=R/'riskbench_social_gan_verified_evaluation_v2/all_seed_events_standalone.csv'
if s.exists():
 g=pd.read_csv(s).rename(columns={'fpr_budget':'budget','scenario_group_id':'family','detected':'hit'});g['hit_by_10']=((g.hit==1)&(g.lead_seconds>=.5-1e-9)).astype(int);parts.append(g)
d=pd.concat(parts,ignore_index=True);rows=[];pairs=[]
for (m,b,seed),g in d.groupby(['method','budget','seed']):
 assert len(g)==420 and g.event_uid.nunique()==420
 pos=g[g.positive==1];neg=g[g.positive==0];rows.append(dict(method=m,budget=b,seed=seed,recall=pos.hit.mean(),false_alarm=neg.hit.mean(),early_05=pos.hit_by_10.mean()))
a=pd.DataFrame(rows);a.to_csv(O/'standalone_per_seed.csv',index=False);a.groupby(['method','budget'])[['recall','false_alarm','early_05']].agg(['mean','std']).to_csv(O/'standalone_summary.csv')
for b in [.05,.1]:
 for m in d.method.unique():
  if m=='rir_all_safe_selected':continue
  for positive,metric in [(1,'hit'),(0,'hit'),(1,'hit_by_10')]:
   z=d[(d.budget==b)&(d.positive==positive)];r=z[z.method=='rir_all_safe_selected'].groupby('family')[metric].mean();v=z[z.method==m].groupby('family')[metric].mean();assert set(r.index)==set(v.index)
   delta=(r-v).to_numpy();rng=np.random.default_rng(20260914);draws=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1);lo,hi=np.quantile(draws,[.025,.975]);pairs.append(dict(budget=b,comparator=m,metric='recall' if metric=='hit' and positive else 'false_alarm' if metric=='hit' else 'early_05',estimate_delta=delta.mean(),ci_low=lo,ci_high=hi))
pd.DataFrame(pairs).to_csv(O/'standalone_paired.csv',index=False)
print(a.groupby(['method','budget'])[['recall','false_alarm','early_05']].mean().to_string());print(pd.DataFrame(pairs).query("comparator=='rollout_ca_shared_risk'").to_string(index=False))
if s.exists():
 P=R.parent/'paper';names={'rollout_ca_shared_risk':'CA rollout','rollout_adaptive_cv_ca_shared_risk':'Adaptive CV/CA rollout','social_gan_safe_only_retrained_shared_risk':'Social-GAN (safe-only retraining)'};out=[]
 for m,label in names.items():
  cells=[]
  for b in [.05,.1]:
   g=a[(a.method==m)&(a.budget==b)]
   for field in ['recall','false_alarm']:
    value=100*g[field].mean();std=100*g[field].std();cells.append(f'${value:.2f}\\pm{std:.2f}$' if len(g)>1 else f'{value:.2f}')
  out.append(label+' & '+' & '.join(cells)+r' \\')
 (P/'strong_standalone_rows.tex').write_text('\n'.join(out)+'\n')
 sg=a[a.method=='social_gan_safe_only_retrained_shared_risk'];v=sg[sg.budget==.1]
 pair=pd.DataFrame(pairs);z=pair[(pair.budget==.1)&(pair.comparator=='social_gan_safe_only_retrained_shared_risk')&(pair.metric=='recall')].iloc[0]
 prose=rf'''Safely retrained Social-GAN reaches ${100*v.recall.mean():.2f}\pm{100*v.recall.std():.2f}$\%
standalone recall and ${100*v.false_alarm.mean():.2f}\pm{100*v.false_alarm.std():.2f}$\% FAR at 10\%.
The RIR-minus-Social-GAN recall interval is [{100*z.ci_low:.2f}, {100*z.ci_high:.2f}]
points. These comparisons use the same full event denominators; the predictors'
history requirements and coverage are documented in the supplement.
'''
 (P/'strong_standalone_comparison_text.tex').write_text(prose)
