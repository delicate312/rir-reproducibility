"""Report all frozen history/geometry comparisons, including negative evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .history_geometry_validation import OUT, METHODS, SNAPSHOT_METHODS, LANDMARKS, match_snapshots
from . import damp_decision_necessity_ablation as d

NBOOT = 10000
RNG_SEED = 20260906


def paired_randomization_p(values, rng):
    """Two-sided sign-flip test; exact enumeration for small nonzero sets."""
    values = np.asarray(values, float)
    values = values[np.abs(values) > 1e-12]
    n = len(values)
    if not n:
        return 1.
    observed = abs(values.sum())
    if n <= 18:
        bits = (np.arange(2**n, dtype=np.uint32)[:, None] >> np.arange(n)) & 1
        sums = (2*bits.astype(float)-1) @ values
        return float(np.mean(np.abs(sums) >= observed-1e-12))
    signs = 2*rng.integers(2, size=(NBOOT, n))-1
    sums = signs @ values
    return float((np.count_nonzero(np.abs(sums) >= observed-1e-12)+1)/(NBOOT+1))


def interval(values, rng):
    values = np.asarray(values, float)
    if not len(values):
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=0)
    boot = values[rng.integers(len(values), size=(NBOOT, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), lo=float(np.quantile(boot, .025)), hi=float(np.quantile(boot, .975)),
                p=paired_randomization_p(values, rng), n=len(values))


def holm(frame, family_col):
    frame = frame.copy()
    frame['p_holm'] = np.nan
    for _, subset in frame.groupby(family_col):
        ss = subset.dropna(subset=['p']).sort_values('p')
        adjusted = np.minimum(1., np.maximum.accumulate(ss.p.to_numpy() * np.arange(len(ss), 0, -1)))
        frame.loc[ss.index, 'p_holm'] = adjusted
    return frame


def matched_analysis(snapshots, availability, rng):
    pairs, coverage = [], []
    for (budget, landmark), block in snapshots.groupby(['budget', 'landmark']):
        for cohort in ('all', 'ttc_negative'):
            for mode in ('geometry', 'cpa_only'):
                reference_pair_ids = None
                for seed, by_seed in block.groupby('seed'):
                    if cohort == 'ttc_negative':
                        by_seed = by_seed[by_seed.ttc_detected.eq(0)]
                    eligible = by_seed[by_seed.cpa_raw.gt(0) & by_seed.damp.notna()]
                    pair_ids = []
                    for fold, subset in eligible.groupby('fold'):
                        a = subset[subset.positive.eq(1)].sort_values('event_uid').reset_index(drop=True)
                        b = subset[subset.positive.eq(0)].sort_values('event_uid').reset_index(drop=True)
                        for i, j, dc, dp, dv, dt, dd in match_snapshots(a, b, mode):
                            p, n = a.iloc[i], b.iloc[j]
                            pair_ids.append((p.event_uid, n.event_uid))
                            for method in SNAPSHOT_METHODS:
                                delta = float(p[method] - n[method])
                                win = .5 if abs(delta) <= 1e-12 else float(delta > 0)
                                pairs.append(dict(budget=budget, landmark=landmark, cohort=cohort, match=mode, seed=seed, fold=fold,
                                    positive_uid=p.event_uid, negative_uid=n.event_uid, method=method, win=win,
                                    positive_score=p[method], negative_score=n[method], delta_cpa=dc, delta_position=dp,
                                    delta_velocity=dv, delta_tcpa=dt, delta_dcpa=dd, positive_actor_is_gt=p.target_is_gt))
                    if reference_pair_ids is None:
                        reference_pair_ids = pair_ids
                    else:
                        assert reference_pair_ids == pair_ids, 'Seed-dependent matching would invalidate paired seed aggregation'
                    aa = availability[(availability.budget.eq(budget)) & (availability.landmark.eq(landmark)) & (availability.seed.eq(seed))]
                    if cohort == 'ttc_negative':
                        aa = aa[aa.ttc_detected.eq(0)]
                    coverage.append(dict(budget=budget, landmark=landmark, cohort=cohort, match=mode, seed=seed,
                        positive_total=int(aa.positive.sum()), negative_total=int((1-aa.positive).sum()),
                        positive_eligible=int(eligible.positive.sum()), negative_eligible=int((1-eligible.positive).sum()), matched_pairs=len(pair_ids)))
    pair_df = pd.DataFrame(pairs)
    pair_df.to_csv(OUT / 'matched_pairs.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT / 'matching_coverage.csv', index=False)
    summaries, comparisons = [], []
    for keys, subset in pair_df.groupby(['budget', 'landmark', 'cohort', 'match']):
        common = dict(zip(['budget', 'landmark', 'cohort', 'match'], keys))
        # Each pair has unique positive/negative events within this landmark;
        # average the three training-seed decisions BEFORE resampling pairs.
        table = subset.groupby(['positive_uid', 'negative_uid', 'method']).win.mean().unstack('method')
        for method in SNAPSHOT_METHODS:
            ci = interval(table[method].to_numpy() - .5, rng)
            summaries.append(dict(common, method=method, n=ci['n'], win_rate=ci['mean']+.5, lo=ci['lo']+.5, hi=ci['hi']+.5))
        for learned in ('damp', 'rir'):
            for baseline in METHODS[:5]:
                ci = interval(table[learned].to_numpy() - table[baseline].to_numpy(), rng)
                comparisons.append(dict(common, learned=learned, baseline=baseline, family='matched', **ci))
    pd.DataFrame(summaries).to_csv(OUT / 'matched_summary.csv', index=False)
    holm(pd.DataFrame(comparisons), 'family').to_csv(OUT / 'matched_statistics.csv', index=False)
    return pd.DataFrame(summaries), pd.DataFrame(coverage), pair_df


def event_analysis(events, rng):
    metrics, contrasts = [], []
    for budget, block in events.groupby('budget'):
        for cohort in ('all_danger', 'hidden_danger', 'ttc_negative_safe'):
            subset = block[block.positive.eq(0 if cohort == 'ttc_negative_safe' else 1)]
            if cohort != 'all_danger':
                subset = subset[subset.ttc_detected.eq(0)]
            for metric in ('hit', 'hit_by_30', 'hit_by_20', 'hit_by_10', 'hit_by_5', 'hit_by_1'):
                if cohort == 'ttc_negative_safe' and metric != 'hit':
                    continue
                table = subset.groupby(['event_uid', 'method'])[metric].mean().unstack('method')
                for method in METHODS:
                    metrics.append(dict(budget=budget, cohort=cohort, metric=metric, method=method, n=len(table), rate=table[method].mean()))
                for baseline in METHODS[:5] + ('rir_no_recovery',):
                    ci = interval(table.rir.to_numpy() - table[baseline].to_numpy(), rng)
                    contrasts.append(dict(budget=budget, cohort=cohort, metric=metric, baseline=baseline,
                        family='false_candidates' if cohort == 'ttc_negative_safe' else 'early_recall', **ci))
    pd.DataFrame(metrics).to_csv(OUT / 'event_summary.csv', index=False)
    holm(pd.DataFrame(contrasts), 'family').to_csv(OUT / 'event_statistics.csv', index=False)
    return pd.DataFrame(metrics)


def release_analysis(releases, rng):
    summaries, comparisons = [], []
    for budget, block in releases.groupby('budget'):
        for method, subset in block.groupby('method'):
            eligible = subset[subset.complete.eq(1)]
            pre = eligible[eligible.pre_alert.eq(1)]
            summaries.append(dict(budget=budget, method=method, total_rows=len(subset), complete_rows=len(eligible), pre_alert_rows=len(pre),
                complete_families=eligible.event_uid.nunique(), pre_alert_families=pre.event_uid.nunique(),
                post_occupancy=pre.post_occupancy.mean(), pre_occupancy=pre.pre_occupancy.mean(),
                release_censored=pre.release_censored.mean(), capped_release_s=pre.release_confirm_s.fillna(1.).mean()))
        for baseline in METHODS[:5] + ('rir_no_recovery',):
            a = block[block.method.eq('rir') & block.complete.eq(1) & block.pre_alert.eq(1)].set_index(['event_uid', 'seed'])
            b = block[block.method.eq(baseline) & block.complete.eq(1) & block.pre_alert.eq(1)].set_index(['event_uid', 'seed'])
            shared = a.index.intersection(b.index)
            for metric in ('post_occupancy', 'pre_occupancy', 'capped_release_s', 'release_censored'):
                av = a.loc[shared, 'release_confirm_s'].fillna(1.) if metric == 'capped_release_s' else a.loc[shared, metric]
                bv = b.loc[shared, 'release_confirm_s'].fillna(1.) if metric == 'capped_release_s' else b.loc[shared, metric]
                diff = (av - bv).groupby('event_uid').mean()
                ci = interval(diff.to_numpy(), rng)
                comparisons.append(dict(budget=budget, baseline=baseline, metric=metric, common_event_seed_n=len(shared), family='release_proxy', **ci))
    pd.DataFrame(summaries).to_csv(OUT / 'release_summary.csv', index=False)
    holm(pd.DataFrame(comparisons), 'family').to_csv(OUT / 'release_statistics.csv', index=False)
    return pd.DataFrame(summaries)


def table_text(frame, columns, percent=()):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in frame.iterrows():
        values = []
        for c in columns:
            v = row[c]
            if pd.isna(v):
                values.append('NA')
            elif c in percent:
                values.append(f'{100*v:.2f}%')
            elif isinstance(v, (float, np.floating)):
                values.append(f'{v:.3f}')
            else:
                values.append(str(v))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def make_plot(matched, event_summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(rir='#b34d45', damp='#b34d45', cpa_peak='#517a99', cpa_ema='#579888', cpa_trend='#b68a3a', cpa_cusum='#8b739e', cpa_temporal='#707c85')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for method in ('damp', 'cpa_peak', 'cpa_ema', 'cpa_trend', 'cpa_cusum'):
        ss = matched[matched.budget.eq(.1) & matched.cohort.eq('all') & matched['match'].eq('geometry') & matched.method.eq(method)].sort_values('landmark')
        axes[0].plot(ss.landmark/20, ss.win_rate*100, 'o-', color=colors[method], label=method)
    axes[0].axhline(50, color='.7', lw=1, ls='--')
    axes[0].set(xlabel='Seconds before critical anchor', ylabel='Matched-pair ranking accuracy (%)', title='Matched present geometry (exploratory)')
    for method in ('rir', 'cpa_peak', 'cpa_ema', 'cpa_trend', 'cpa_cusum'):
        ss = event_summary[event_summary.budget.eq(.1) & event_summary.cohort.eq('all_danger') & event_summary.method.eq(method) & event_summary.metric.isin(['hit_by_20', 'hit_by_10', 'hit_by_5'])].copy()
        ss['seconds'] = ss.metric.str.removeprefix('hit_by_').astype(int)/20
        ss = ss.sort_values('seconds')
        axes[1].plot(ss.seconds, ss.rate*100, 'o-', color=colors[method], label=method)
    axes[1].set(xlabel='Minimum lead before critical anchor (s)', ylabel='Recall over all 147 danger families (%)', title='10% safe-validation budget (channel only)')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.15)
    fig.text(.5, .01, 'Existing internal families; matched sample sizes and uncertainty are reported in the accompanying tables.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(OUT / 'history_geometry_diagnostic.png', dpi=180)
    fig.savefig(OUT / 'history_geometry_diagnostic.svg')
    plt.close(fig)


def main():
    # Freeze analysis definitions before reading score outputs.
    analysis_protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=d.sha256_file(Path(__file__)),
        bootstrap=NBOOT, rng_seed=RNG_SEED, matched_unit='one uniquely matched positive/negative event pair at a fixed landmark; average seed win indicators first',
        p_values='paired sign-flip randomization; exact for <=18 nonzero differences, otherwise 10000 Monte Carlo flips; exchangeability null; small-sample correction replaces archived bootstrap-tail p values',
        event_unit='event family; seed decisions averaged before resampling',
        multiplicity='Holm across all contrasts within matched, early_recall, false_candidates, or release_proxy families; intervals pointwise',
        release_unit='family means over event-seed rows where BOTH methods alerted in final 0.5s before behavior end; selection conditional and exploratory',
        release_censoring='unresolved within 1 s capped at 1 s plus censoring rate; not an uncensored mean release time',
        statistics_scope='conditional on frozen predictions and matched sample; does not refit models or account for previous development reuse')
    p = OUT / 'analysis_protocol.json'
    if p.exists() and json.loads(p.read_text())['source_sha256'] != analysis_protocol['source_sha256']:
        raise RuntimeError('Analysis changed after freeze; preserve previous outputs first')
    if not p.exists():
        p.write_text(json.dumps(analysis_protocol, indent=2))
    rng = np.random.default_rng(RNG_SEED)
    snapshots = pd.read_csv(OUT / 'snapshots.csv')
    availability = pd.read_csv(OUT / 'availability.csv')
    events = pd.read_csv(OUT / 'events.csv')
    releases = pd.read_csv(OUT / 'release_proxy.csv')
    assert events.groupby(['budget', 'seed', 'method']).size().eq(420).all()
    assert events[events.positive.eq(1)].groupby(['budget', 'seed', 'method']).size().eq(147).all()
    matched, coverage, pairs = matched_analysis(snapshots, availability, rng)
    event_summary = event_analysis(events, rng)
    release_summary = release_analysis(releases, rng)
    make_plot(matched, event_summary)
    lines = ['# 当前几何匹配、历史增量与撤警诊断', '',
        '本次使用现有已审阅 RiskBench 家族、五折和三个训练种子；没有训练新模型，也没有选择最优方法替换正文。结果属于内部探索性验证。', '',
        '## 固定设计与可解释范围', '',
        '- 1.0 / 0.5 / 0.25 秒前的固定快照；只用当时及过去观测。当前最高 CPA 对象由几何选择，GT 风险目标只用于事后核查。',
        '- 主要匹配同时限制位置、速度、TCPA、DCPA 与 CPA 分数；较弱的 CPA 分数匹配完整保留。每个时刻、折和队列内不重复使用事件。',
        '- 排序准确率表示一对匹配事件中，危险事件所选对象的分数高于安全事件所选对象的比例，并列记 0.5；它不是事件召回率，也不是严格因果效应。',
        '- 匹配不能证明安全/危险事件只在历史上不同；场景和其他参与者仍可能不同。覆盖率低时不能外推到所有场景。',
        '- 每种历史基线独立用 hard-safe 验证数据定阈值；5%/10% 是验证约束，不是测试集候选率保证。历史基线是固定简单控制，并非穷尽了最佳物理预测器。',
        '- 原论文 RIR、单帧 CPA、三帧 CPA 在两种预算全部 15 次运行逐事件分数、阈值与 TTC 分数复现，误差不超过 1e-10。',
        '- 提前量主表是各独立通道的检出，不含 TTC 并集；all_danger 分母 147，hidden_danger 分母 33。没有对检出成功的子集单独比较平均提前量。',
        '- 已有预测器/组件保留原来的因果预热；新 CPA 历史头从正式 30 帧窗口开始。这不是对神经历史编码器的直接干预消融。', '',
        '## 匹配覆盖率（每个种子相同；展示 10% 版本）', '']
    cv = coverage[coverage.budget.eq(.1) & coverage.seed.eq(d.DEFAULT_SEEDS[0])]
    lines.append(table_text(cv, ['cohort', 'match', 'landmark', 'positive_total', 'positive_eligible', 'negative_eligible', 'matched_pairs']))
    lines += ['', '## 严格几何匹配后的排序（全体事件，10% 版本）', '']
    mm = matched[matched.budget.eq(.1) & matched.cohort.eq('all') & matched['match'].eq('geometry')]
    lines.append(table_text(mm, ['landmark', 'method', 'n', 'win_rate', 'lo', 'hi'], percent=['win_rate', 'lo', 'hi']))
    lines += ['', '全部方法、两个预算、两种匹配及两个队列保存在 matched_summary.csv；差值区间和 Holm 值在 matched_statistics.csv。', '',
        '## 相同验证预算下的事件与提前量表现', '']
    for budget in (.05, .1):
        lines += ['', f'### 验证预算 {budget:.0%}', '']
        es = event_summary[event_summary.budget.eq(budget)]
        rows = []
        for method in METHODS:
            def value(cohort, metric):
                return es[(es.method.eq(method)) & es.cohort.eq(cohort) & es.metric.eq(metric)].rate.iloc[0]
            rows.append(dict(method=method, hidden_recall=value('hidden_danger', 'hit'), added_rate=value('ttc_negative_safe', 'hit'),
                all_by_1s=value('all_danger', 'hit_by_20'), all_by_0p5s=value('all_danger', 'hit_by_10'), all_by_0p25s=value('all_danger', 'hit_by_5')))
        lines.append(table_text(pd.DataFrame(rows), list(rows[0]), percent=list(rows[0])[1:]))
    lines += ['', '## 撤警：仅行为结束后的诊断', '',
        '现有 metadata 的 behavior_window 不是经过独立审定的危险起止区间。以下只描述官方行为结束后的告警滞留，不将这些时刻改成新的安全/危险真值。',
        '要求行为结束前 0.5 秒及后 1 秒完整可用。每个方法的摘要限于其结束前已告警事件；方法差值另用双方共同告警的事件-种子交集，并按家族 bootstrap。',
        '连续五帧低于阈值才确认撤警；1 秒内未完成记删失，封顶时间只能解释为有限观察窗的诊断量。较短告警也可能来自较弱的风险响应，必须结合 pre_occupancy 和共同检出样本判断。', '']
    lines.append(table_text(release_summary, ['budget', 'method', 'complete_families', 'pre_alert_families', 'pre_occupancy', 'post_occupancy', 'release_censored', 'capped_release_s'], percent=['pre_occupancy', 'post_occupancy', 'release_censored']))
    lines += ['', '## 后续确认所需证据', '',
        '1. 独立的新家族或预先定义的成对场景；不能继续用现有 33 个隐藏危险选择最优权重。',
        '2. 隐藏模型分数的风险形成/解除标注，并保留未碰撞但需要避让的场景。',
        '3. 对历史预测器进行同输入、同训练协议的直接消融；本次只检验冻结输出，不证明历史编码本身的因果贡献。',
        '4. 用更长连续安全片段校准和评价误报警次数/小时；现有片段不能提供该部署指标。', '',
        '复现：先运行 history_geometry_validation，再运行 history_geometry_report；使用 immune_driving Python 环境，numpy 在 torch 之前导入。']
    (OUT / 'HISTORY_GEOMETRY_REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT / 'analysis_completion.json').write_text(json.dumps(dict(completed_utc=datetime.now(timezone.utc).isoformat(), source_sha256=d.sha256_file(Path(__file__)), matched_rows=len(pairs), outputs='matched/event/release summaries and paired statistics; scientific diagnostic PNG/SVG'), indent=2))
    print('Report complete:', OUT / 'HISTORY_GEOMETRY_REPORT.md', flush=True)


if __name__ == '__main__':
    main()
