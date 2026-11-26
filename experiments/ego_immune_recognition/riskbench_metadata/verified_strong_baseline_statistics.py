"""Final three-training-seed baseline alignment and manuscript table export."""
from pathlib import Path
import numpy as np,pandas as pd,json
from . import selected_strong_baseline_statistics as a
from .mainstream_baseline_evaluation_v2 import _summarize_seed,_aggregate_seed_summary
R=a.RESULTS;O=R/'riskbench_verified_strong_baselines_v2';O.mkdir(exist_ok=True)
a.COMPARATOR_FILES=(R/'riskbench_actor_query_transformer_v1/all_seed_events.csv',R/'riskbench_probabilistic_rollout_v2/all_seed_events.csv',R/'riskbench_social_gan_verified_evaluation_v2/all_seed_events.csv');a.OUTPUT=O
a.run(replicates=10000)
data=pd.concat([a._load_budget(b) for b in [.05,.1]],ignore_index=True)
assert np.isfinite(data.score.to_numpy(float)).all()
for (m,b,seed),g in data.groupby(['method','fpr_budget','seed']):
 assert len(g)==420 and g.event_uid.nunique()==420 and g.scenario_group_id.nunique()==420
 assert int(g.positive.sum())==147 and int(((g.positive==1)&(g.ttc_detected==0)).sum())==33
 assert int(((g.positive==0)&(g.ttc_detected==0)).sum())==216
 assert np.array_equal(g.final_detected.astype(bool),(g.detected.astype(bool)|g.ttc_detected.astype(bool)))
 assert np.array_equal(g.blind_rescue.astype(bool),(g.detected.astype(bool)&~g.ttc_detected.astype(bool)))
records=data.replace({np.nan:None}).to_dict('records');seed_summary=_summarize_seed(records);agg=pd.DataFrame(_aggregate_seed_summary(seed_summary));pd.DataFrame(seed_summary).to_csv(O/'per_seed_summary.csv',index=False);agg.to_csv(O/'aggregate.csv',index=False)
sg=pd.read_csv(a.COMPARATOR_FILES[-1]);assert sg.training_seed.nunique()==3
coverage=sg[sg.fpr_budget==.05].groupby(['seed','positive']).agg(events=('event_uid','size'),events_with_scores=('observed_score_frames',lambda x:int((x>0).sum())),target_history_available=('target_history_available','sum')).reset_index();coverage.to_csv(O/'coverage.csv',index=False)
thresholds=pd.read_csv(R/'riskbench_social_gan_verified_evaluation_v2/all_seed_thresholds.csv');assert (thresholds.validation_incremental_fpr<=thresholds.fpr_budget+1e-12).all()
P=a.PROJECT_ROOT/'paper'
names={'rollout_ca_shared_risk':'CA rollout','rollout_adaptive_cv_ca_shared_risk':'Adaptive CV/CA rollout','social_gan_safe_only_retrained_shared_risk':'Social-GAN (safe-only retraining)','supervised_actor_query_transformer':'Actor-query Transformer','rir_nested_selected':'RIR'}
metrics=['overall_recall','blind_rescue_recall','incremental_rescue_fpr','mean_blind_rescue_lead_seconds','blind_auroc','blind_average_precision']
def cells(method,b,fields=metrics):
 row=agg[(agg.method==method)&(agg.fpr_budget==b)].iloc[0];v=[]
 for field in fields:
  scale=100 if field in metrics[:3] else 1;dec=2 if scale==100 else 3;mean=row[field+'_mean']*scale;std=row[field+'_std']*scale
  v.append(f'${mean:.{dec}f}\\pm{std:.{dec}f}$' if method in ['social_gan_safe_only_retrained_shared_risk','supervised_actor_query_transformer','rir_nested_selected'] else f'{mean:.{dec}f}')
 return v
methods=list(names)[:3]
(P/'strong_baseline_rows.tex').write_text('\n'.join(names[m]+' & '+' & '.join(cells(m,.05))+r' \\' for m in methods)+'\n')
table=r'''\begin{table}[t]
\centering
\caption{Additional forecasting comparisons at the 10\% validation additional-candidate budget. All rows combine the physical reference with the named method. Learned rows report mean $\pm$ sample standard deviation over three independent training seeds.}
\label{tab:forecast_review}
\small
\begin{tabularx}{\textwidth}{XCCC}
\toprule
Method & Recall (\%) & Hidden discovery (\%) & Added candidates (\%) \\
\midrule
'''
for m in methods+['rir_nested_selected']:table+=names[m]+' & '+' & '.join(cells(m,.1,metrics[:3]))+r' \\'+'\n'
table+='\\bottomrule\n\\end{tabularx}\n\\end{table}\n';(P/'strong_baseline_review_table.tex').write_text(table)
print(agg[['method','fpr_budget','overall_recall_mean','overall_recall_std','blind_rescue_recall_mean','incremental_rescue_fpr_mean']].to_string(index=False));print(coverage.to_string(index=False))
(O/'completion.json').write_text(json.dumps({'status':'complete','bootstrap_replicates':10000,'actual_social_gan_training_seeds':3,'event_contract_checks':'passed'}))
