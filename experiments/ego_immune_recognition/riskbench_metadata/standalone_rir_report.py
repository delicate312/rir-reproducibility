"""Full-population reporting for RIR without a TTC alarm channel."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .standalone_rir_audit import OUT, HISTORY_OUT
from .history_geometry_report import interval, holm, table_text
from .damp_incremental_value_statistics import weighted_curve_metrics
from . import damp_decision_necessity_ablation as d


def main():
    (OUT/'analysis_protocol.json').write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=d.sha256_file(Path(__file__)),
        unit='scenario family; not frames or training seeds; 147 danger and 273 hard-safe',
        uncertainty='10000 paired family bootstrap; binary seed decisions averaged before resampling; exact/Monte Carlo paired sign-flip p; Holm across all contrasts and both budgets per binary metric',
        curves='AP/AUROC/pAUC10 computed within each seed then averaged; identical stratified family bootstrap weights for all methods; contrast intervals only, no unvalidated bootstrap-tail significance claims',
        comparisons='fresh all-safe RIR against every other recalibrated method; original-head recalibrated RIR against six geometry/content baselines',
        status='exploratory on previously examined internal families; no winner selected'),indent=2))
    x=pd.read_csv(OUT/'events.csv')
    thresholds=pd.read_csv(OUT/'thresholds.csv')
    assert thresholds.val_rate.le(thresholds.budget+1e-12).all()
    assert x.groupby(['regime','budget','seed','method']).size().eq(420).all()
    assert x[x.positive.eq(1)].groupby(['regime','budget','seed','method']).size().eq(147).all()
    seeds=[]
    for (regime,budget,method,seed),ss in x.groupby(['regime','budget','method','seed']):
        pos=ss[ss.positive.eq(1)];safe=ss[ss.positive.eq(0)]
        tp=int(pos.hit.sum());fp=int(safe.hit.sum())
        row=dict(regime=regime,budget=budget,method=method,seed=seed,recall=pos.hit.mean(),hard_safe_fpr=safe.hit.mean(),
            precision=tp/(tp+fp) if tp+fp else np.nan,detected=tp,false_alarms=fp,AP=average_precision_score(ss.positive,ss.score),
            AUROC=roc_auc_score(ss.positive,ss.score),pAUC10=roc_auc_score(ss.positive,ss.score,max_fpr=.1))
        for h in (20,10,5):
            row[f'recall_by_{h}']=pos[f'hit_by_{h}'].mean()
        seeds.append(row)
    seeds=pd.DataFrame(seeds);seeds.to_csv(OUT/'seed_metrics.csv',index=False)
    metrics=['recall','hard_safe_fpr','precision','AP','AUROC','pAUC10','recall_by_20','recall_by_10','recall_by_5']
    summary=seeds.groupby(['regime','budget','method'])[metrics].mean().reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    rng=np.random.default_rng(20260908)
    controls=('cpa_peak','cpa_temporal','cpa_ema_warmup','cpa_cusum_warmup','constant_damp_control','raw_residual_hybrid')
    recal=x[x.regime.eq('all_safe_recalibrated')]
    contrasts=[]
    for budget,ss in recal.groupby('budget'):
        for metric,positive,column in [('recall',1,'hit'),('hard_safe_fpr',0,'hit'),('recall_by_20',1,'hit_by_20'),('recall_by_10',1,'hit_by_10'),('recall_by_5',1,'hit_by_5')]:
            table=ss[ss.positive.eq(positive)].groupby(['family','method'])[column].mean().unstack('method')
            pairs=[('rir_all_safe_selected',m) for m in table.columns if m!='rir_all_safe_selected']+[('rir_original_head',m) for m in controls]
            for method,baseline in pairs:
                ci=interval(table[method].to_numpy()-table[baseline].to_numpy(),rng)
                contrasts.append(dict(budget=budget,metric=metric,method=method,baseline=baseline,family=metric,**ci))
    holm(pd.DataFrame(contrasts),'family').to_csv(OUT/'paired_statistics.csv',index=False)
    # Rank-metric intervals use the same family resampling across all seeds.
    unique=recal[['family','positive']].drop_duplicates().sort_values('family')
    y=unique.positive.to_numpy(int)
    assert len(unique)==420 and y.sum()==147
    weights=np.ones((10001,len(y)),dtype=np.int16)
    for klass,n in ((1,147),(0,273)):
        weights[1:,y==klass]=rng.multinomial(n,np.full(n,1/n),size=10000)
    dist={}
    for (budget,method),ss in recal.groupby(['budget','method']):
        arrays=[]
        for seed,sub in ss.groupby('seed'):
            sub=sub.set_index('family').loc[unique.family]
            values=weighted_curve_metrics(y,sub.score.to_numpy(),weights)
            np.testing.assert_allclose(values['AP'][0],average_precision_score(y,sub.score),atol=1e-12)
            np.testing.assert_allclose(values['AUROC'][0],roc_auc_score(y,sub.score),atol=1e-12)
            arrays.append(values)
        dist[(budget,method)]={k:np.mean([a[k] for a in arrays],axis=0) for k in ('AP','AUROC','pAUC10')}
    curves=[]
    for budget in (.05,.1):
        for method in ('rir_all_safe_selected','rir_original_head'):
            for baseline in controls:
                for metric in ('AP','AUROC','pAUC10'):
                    delta=dist[(budget,method)][metric]-dist[(budget,baseline)][metric]
                    curves.append(dict(budget=budget,method=method,baseline=baseline,metric=metric,difference=delta[0],lo=np.quantile(delta[1:],.025),hi=np.quantile(delta[1:],.975)))
    pd.DataFrame(curves).to_csv(OUT/'ranking_intervals.csv',index=False)
    # A matched event-level decomposition describes what removing TTC loses.
    old=pd.read_csv(HISTORY_OUT/'events.csv');old=old[old.method.eq('rir')]
    decomp=[]
    for (budget,seed,positive),ss in old.groupby(['budget','seed','positive']):
        decomp.append(dict(budget=budget,seed=seed,positive=positive,n=len(ss),
            both=int((ss.hit.eq(1)&ss.ttc_detected.eq(1)).sum()),rir_only=int((ss.hit.eq(1)&ss.ttc_detected.eq(0)).sum()),
            ttc_only=int((ss.hit.eq(0)&ss.ttc_detected.eq(1)).sum()),neither=int((ss.hit.eq(0)&ss.ttc_detected.eq(0)).sum())))
    pd.DataFrame(decomp).to_csv(OUT/'frozen_channel_decomposition.csv',index=False)
    # Explicitly retain empty-window events at zero instead of excluding them.
    coverage=[]
    for fold in range(5):
        frames=d._load_frames(d.DEFAULT_SOURCE_ROOT,d.DEFAULT_SEEDS[0],fold)
        for split in ('validation_normal','test_normal','test_positive'):
            events=d._event_frames(frames[split])
            coverage.append(dict(fold=fold,split=split,n=len(events),empty_pre_anchor=sum(not d._usable_frames(fs) for fs in events.values())))
    pd.DataFrame(coverage).to_csv(OUT/'window_coverage.csv',index=False)
    selection=pd.read_csv(OUT/'selected_parameters.csv')
    lines=['# 无 TTC 报警兜底的 RIR：独立检测验证','',
        '本次回答的是去掉 TTC 报警并集，以及不再按 TTC 阴性子集校准/选择参数后的表现。原有模型仍使用 inverse TTC 关系特征，并保留 CPA 对齐与极端 CPA 旁路；不是彻底去掉 TTC 特征或所有物理信息后的重训练结果。','',
        '## 三种实验不可混用','',
        '1. **直接移除 TTC 并集**：原模型、DAMP 系数和阈值不变，只看 RIR 独立报警。回答当前系统依赖 TTC 兜底多少。',
        '2. **全部 hard-safe 重新校准**：原系数不变，阈值仅用全部 validation_normal 事件校准到 5%/10%。旧系数仍有历史上的 TTC 条件选择依赖。',
        '3. **全部 hard-safe 重新选择和校准**：同一既有 90 候选网格、五个内部家族子折，仅依赖全部 hard-safe 的校准稳定性；不使用 TTC 判决划分事件，不使用危险标签选择头或阈值。另保留两个既有固定系数头。',
        '所有操作点最终在全部 147 个危险、273 个 hard-safe 家族上评价。误报率分母均为 273，不能与论文中以 216 为分母的 added-candidate rate 混用。','',
        '## 直接去掉 TTC：原阈值','',
        table_text(summary[summary.regime.eq('remove_union_only')],['budget','recall','hard_safe_fpr','recall_by_20','recall_by_10','recall_by_5'],percent=['recall','hard_safe_fpr','recall_by_20','recall_by_10','recall_by_5']),'',
        '## 公平的独立检测比较：全部 hard-safe 校准','']
    for budget in (.05,.1):
        ss=summary[summary.regime.eq('all_safe_recalibrated')&summary.budget.eq(budget)]
        lines += [f'### {budget:.0%} 验证预算','',table_text(ss,['method','recall','hard_safe_fpr','recall_by_20','recall_by_10','recall_by_5','AP'],percent=['recall','hard_safe_fpr','recall_by_20','recall_by_10','recall_by_5']),'']
    lines += ['方法名：rir_original_head 是旧系数重校准；rir_all_safe_selected 是全量安全数据重新选择；rir_fixed_simple / rir_fixed_paper 是既有固定参数，未根据这次外部效果挑选。constant_damp_control 保留融合及三帧规则但用常数代替 DAMP；raw_residual_hybrid 使用同一预测器的原始残差。','',
        '## 审计与限制','',
        f'- 新参数选择共 {len(selection)} 个折/种子/预算配置，内部校准无候选满足预算而采用既有 fallback 的次数为 {int(selection.fallback.sum())}。完整最终验证集上的所有阈值均满足各自预算；测试误报率仍可能偏离预算。',
        '- 向量化安全选择器在真实数据上与直接因果评分复现至 1e-10；原 RIR 和六个控制的外部事件分数也与此前缓存结果一致。',
        '- 无可用预关键点帧的事件保留为零分，不从分母排除；数量见 window_coverage.csv。',
        '- 固定已有预测器，仅调整决策头选择/校准。没有进行完全移除 inverse TTC 输入与预测目标的重训练，不能宣称 feature-level TTC-free。',
        '- CPA 阈值与 RIR 阈值各自只在同一全量 hard-safe 验证集确定，不能用测试危险标签把两个方法调到更有利的点。',
        '- 精确率和 AP 只适用于当前病例组成；不能当作实际道路危险发生率下的审阅收益。',
        '- 提前量相对官方 critical anchor，所有危险都在分母中；没有仅比较各自检出子集的平均提前量。',
        '- 这些家族已用于多轮开发，所有结果仍为内部探索性证据。',
        '- paired_statistics.csv 给出家族配对差值、点态区间和每个指标内的 Holm 校正；ranking_intervals.csv 给出 AP/AUROC/pAUC 的配对区间。','',
        '复现：standalone_rir_audit → standalone_rir_report；使用 immune_driving Python，numpy 在 torch 前导入。']
    (OUT/'STANDALONE_RIR_REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    colors={'rir_all_safe_selected':'#b34d45','rir_original_head':'#d48770','cpa_peak':'#517a99','cpa_cusum_warmup':'#579888','constant_damp_control':'#a58a53','raw_residual_hybrid':'#8b739e'}
    labels={'rir_all_safe_selected':'RIR: all-safe selection','rir_original_head':'RIR: original head','cpa_peak':'Peak CPA','cpa_cusum_warmup':'CPA accumulator','constant_damp_control':'Constant DAMP control','raw_residual_hybrid':'Raw residual hybrid'}
    for ax,budget in zip(axes,(.05,.1)):
        ss=summary[summary.regime.eq('all_safe_recalibrated')&summary.budget.eq(budget)]
        for method,color in colors.items():
            row=ss[ss.method.eq(method)].iloc[0]
            ax.scatter(row.hard_safe_fpr*100,row.recall*100,color=color,label=labels[method],s=48)
        ax.axvline(budget*100,color='.7',ls='--',lw=1)
        ax.set(title=f'{budget:.0%} all-hard-safe validation budget',xlabel='Test hard-safe false-alarm rate (%)',ylabel='Recall over 147 danger families (%)')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15)
    handles,labels_list=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels_list,loc='lower center',ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,.14,1,1))
    fig.savefig(OUT/'standalone_tradeoff.png',dpi=180);fig.savefig(OUT/'standalone_tradeoff.svg')
    plt.close(fig)
    (OUT/'analysis_completion.json').write_text(json.dumps(dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=d.sha256_file(Path(__file__)),rows=len(x)),indent=2))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
