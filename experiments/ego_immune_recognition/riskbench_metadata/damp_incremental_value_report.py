"""Create a reviewable evidence report and plots; does not edit the manuscript."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/ego_immune_riskbench_damp_incremental_value_20260906"
LABELS = {
    "current": "Current RIR", "cpa_peak": "CPA: single-frame peak",
    "cpa_temporal": "CPA: same temporal rules", "constant_damp_control": "DAMP replaced by constant",
    "conditional_median_control": "DAMP replaced by safe median", "raw_residual_hybrid": "Raw residual + same geometry",
    "cpa_weighted": "75% CPA / 25% DAMP", "conditional_damp": "Conditional DAMP calibration",
    "joint_tail": "Joint-tail fusion", "two_of_five": "Two observations within five",
}
ORDER = list(LABELS)
GREEN, BLUE, GRAY, ORANGE = "#1b8a78", "#426d9b", "#87979c", "#b67d26"


def main():
    metrics = pd.read_csv(OUT / "seed_metrics.csv")
    means = metrics.groupby(["budget", "method"]).mean(numeric_only=True)
    stds = metrics.groupby(["budget", "method"]).std(numeric_only=True)
    stats = pd.read_csv(OUT / "paired_statistics.csv")
    ranking = pd.read_csv(OUT / "fixed_review_ranking.csv").groupby(["budget", "method", "k"]).mean(numeric_only=True)
    event_rows = pd.read_csv(OUT / "events.csv")
    conservative = event_rows[event_rows.budget.eq(.05) & event_rows.ttc_detected.eq(0)]
    disagreements = conservative[conservative.method.eq("current")].merge(
        conservative[conservative.method.eq("constant_damp_control")],
        on=["seed", "fold", "family"], suffixes=("_rir", "_control"))
    disagreements[["seed", "fold", "family", "event_uid_rir", "split_rir", "target_type_rir",
        "detected_rir", "detected_control", "score_rir", "threshold_rir", "score_control", "threshold_control"]].to_csv(
        OUT / "mechanism_disagreements.csv", index=False)
    # Direct cost-vs-recall display prevents treating larger queues as free gains.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "axes.edgecolor": "#aab5b8"})
    figure = plt.figure(figsize=(11, 4.9))
    axes = [figure.add_axes([.25, .15, .235, .67]),
            figure.add_axes([.765, .15, .205, .67])]
    groups = [
        ["cpa_peak", "cpa_temporal", "constant_damp_control", "raw_residual_hybrid", "current"],
        ["cpa_peak", "current", "cpa_weighted", "conditional_damp", "joint_tail", "two_of_five"],
    ]
    for axis, budget, names, title in zip(axes, [.05, .10], groups,
        ["(a) Mechanism controls: 5% budget", "(b) New heads: 10% budget"]):
        for row, name in enumerate(names):
            m = means.loc[budget, name]
            s = stds.loc[budget, name]
            color = GREEN if name == "current" else BLUE if name.startswith("cpa_") and name != "cpa_weighted" else ORANGE if name == "cpa_weighted" else GRAY
            axis.errorbar(m.hidden_recall, row, xerr=s.hidden_recall, fmt="o", color=color,
                          markersize=7 if name in ("current", "cpa_weighted") else 5, capsize=3, lw=1.1)
            axis.text(69, row, f"{m.added_rate:.2f}%", color=color, fontsize=8.5, va="center", ha="center", clip_on=False)
            axis.annotate(f"{m.hidden_recall:.1f}", (m.hidden_recall, row), xytext=(0, 10),
                          textcoords="offset points", ha="center", color=color, fontsize=8)
        axis.set_yticks(range(len(names)), [LABELS[name] for name in names], fontsize=8)
        axis.set_ylim(len(names)-.4, -.7)
        axis.set_xlim(15, 65)
        axis.set_xticks([20, 30, 40, 50, 60])
        axis.set_xlabel("Hidden-danger recall (%)")
        axis.set_title(title, fontsize=9.5, weight="bold", loc="left", pad=23)
        axis.text(69, -.8, "Added\nhard-safe", ha="center", va="bottom", fontsize=8, color="#66767c", clip_on=False)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", color="#d8e0e2", linewidth=.6)
        axis.set_axisbelow(True)
    figure.suptitle("DAMP contribution and the cost of additional discovery", x=.025, ha="left", fontsize=13, weight="bold", y=.99)
    figure.text(.025, .922, "Exploratory internal audit | five family folds, three seeds | each threshold fitted on hard-safe validation only", fontsize=8.6, color="#66767c")
    figure.text(.025, .035, "Points: three-seed mean; whiskers: seed SD. Recall denominator: 33 dangers missed by TTC; cost denominator: 216 hard-safe events.", fontsize=8, color="#66767c")
    for extension in ("png", "svg"):
        figure.savefig(OUT / f"damp_evidence_audit.{extension}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    table = []
    for name in ORDER:
        a, b = means.loc[.05, name], means.loc[.10, name]
        table.append(f"| {LABELS[name]} | {a.hidden_recall:.2f}% | {a.added_rate:.2f}% | {b.hidden_recall:.2f}% | {b.added_rate:.2f}% |")
    ranked = []
    for name in ("cpa_peak", "cpa_temporal", "current", "cpa_weighted"):
        ranked.append(f"| {LABELS[name]} | " + " | ".join(f"{ranking.loc[(.05,name,k),'expected_hits']:.2f}" for k in (10,25,50)) + " |")
    evidence = []
    for budget, reference in [(.05,"cpa_peak"),(.05,"cpa_temporal"),(.05,"constant_damp_control"),(.05,"raw_residual_hybrid"),(.10,"cpa_peak"),(.10,"raw_residual_hybrid")]:
        row=stats[(stats.method=="current")&(stats.reference==reference)&(stats.budget==budget)&(stats.metric=="hidden_recall")].iloc[0]
        evidence.append(f"| {int(budget*100)}% | {LABELS[reference]} | {row.difference:+.2f} | [{row.ci95_low:.2f}, {row.ci95_high:.2f}] | {row.p_unadjusted:.4f} | {row.p_holm:.4f} |")

    report = """# DAMP 相对 CPA 的增量价值：机制对照与改进实验

日期：2026-09-06。状态：已有内部家族上的探索性开发证据；没有选择“赢家”替换正式方法，也没有修改论文结果。

## 结论

“单帧 CPA 在 10% 审阅预算下已经很好”是一个真实而重要的对照结果，但不能据此推出 DAMP 没有作用。本次最有力的证据来自原先的 **5% 保守预算**：在保留相同几何变换、同一目标三帧汇总和极端 CPA 旁路时，真实 DAMP 比移除 DAMP 的控制多恢复 11.11 个百分点的危险，额外候选率还更低。该机制对比经本轮多重比较校正仍有统计支持。

但这种结果不等于全面优于最强 CPA 使用方式。对单帧取峰值的 CPA，正式 RIR 在两种预算下的召回优势都尚无充分统计证据。排序、低 FPR 区域和时效呈局部改善，不能作为全面优势来宣传。

四个固定改进头中，75% CPA / 25% DAMP 融合最值得继续验证：在 10% 验证预算下达到 60.61% hidden recall、9.10% added hard-safe rate、91.16% 总体召回。但相比正式 RIR 的 57.58% / 8.02%，提升不显著，而且代价增加。不能仅凭新数字替换摘要。

## 1. 实验与验证

- 固定原始五折、三个训练种子、检查点、因果观测及 30 帧评估窗口。
- 两个预算分别复用原来只从 hard-safe 验证数据选择的 DAMP 系数。每个控制/改进头独立用 TTC-negative validation_normal 校准事件阈值，然后评分 outer test。
- 在运行前写出 `protocol.json`，固定全部十种方式；不根据外部正例结果调整参数，不选择报告中的最好一行。
- 三个正式控制（RIR、单帧 CPA、时间汇总 CPA）在两种预算、全部 15 次运行中，逐事件分数、阈值与正式结果误差小于 1e-10，检测标志相符。
- 共生成 25,200 条外部事件预测和 300 条阈值记录；所有完整验证集操作点均满足各自预算。测试候选率不是保证值，可能超过预算。
- 联合尾概率包含等值样本并作 +1 平滑；单元测试与暴力计数一致。时间汇总验证了同一目标约束及前缀因果性。向量化 bootstrap 指标与 sklearn 的 AP、AUROC 和标准化 partial AUROC 数值相符。

## 2. 全部结果

左侧两个数字使用 5% 验证预算，右侧使用 10%；每列是三次训练的均值。恢复率分母为 33 个 TTC-negative 危险家族，新增候选率分母为 216 个 TTC-negative hard-safe 家族。

| 方式 | 5%：恢复率 | 5%：新增候选率 | 10%：恢复率 | 10%：新增候选率 |
|---|---:|---:|---:|---:|
""" + "\n".join(table) + """

控制含义：

- `cpa_peak` 是已有最简单 CPA 比较方式，直接取窗口内峰值，保留其所有可用几何观测。
- `cpa_temporal` 加上与 RIR 对应的三帧汇总与极端 CPA 旁路。
- `constant_damp_control` 将 DAMP 分位数固定为 1，保留原来的平方根融合、时间规则与关系可用性；因此是隔离真实 DAMP 内容的更直接控制。
- `conditional_median_control` 以验证集中相近 CPA 范围下的典型 DAMP 代替当前事件的真实 DAMP，只保留正常数据中的几何相关性。其 10% 点也能达到 60.61%，但新增候选率为 10.03%，AP 明显较低。这再次说明较高单点召回本身不能证明 DAMP 的价值。
- `raw_residual_hybrid` 使用同一安全预测器的原始预测误差分位数，配上同样的几何和时间规则；它测试的是 DAMP 变换相对“只是预测误差”的价值。

四项改进固定为：CPA 权重从 0.5 改为 0.75；在固定 CPA 分层内校准 DAMP；安全数据上的二维联合尾概率；五个时间位置内取同一目标最高两次证据的均值。CPA 分层固定为 [0,.8)、[.8,.95)、[.95,.99)、[.99,1]，不足 100 条参考关系时退回全局参考。这些是按机制提出的固定原型，未作基于正例的搜索。

## 3. 哪条证据能支持论文

采用 10,000 次配对家族 bootstrap。一次抽样对所有方法和种子使用相同家族权重；每次先算各个种子的指标，再取三种子均值。正负家族分层重采样，维持 33/216 的总体组成。

机制控制 family 包含当前 RIR 对五个控制、两个预算，共十个对比；对每个指标单独进行 Holm 校正。原型 family 包含四个新头对当前 RIR、两个预算，共八个对比。以下区间为未做同时覆盖校正的 95% 区间，p_Holm 为该指标比较 family 内的校正值。所有推断仍以已见内部数据上的探索性诊断解释，不能当作新外部确认。

| 验证预算 | RIR 对照方式 | 恢复率差值 / pp | 95% 配对区间 | p | p_Holm |
|---|---|---:|---|---:|---:|
""" + "\n".join(evidence) + """

最值得保留的是保守预算下的机制对照：

1. 真实 DAMP：44.44% 恢复率、3.24% 新增候选；同结构去掉 DAMP：33.33%、4.63%。增益并非靠更多候选换来。校正后 p≈0.036。
2. 将 DAMP 换为同一模型的原始预测误差后：30.30%、2.78%。恢复率差异校正后 p≈0.016，但这个对照候选率也略低，不能把它单独解释成完全相同测试代价下的优势。
3. 对未加时间汇总的单帧 CPA，差异不确定；必须同时保留这一强对照，不能只展示更有利的时间匹配 CPA。

逐事件核查也有助于解释：相对 constant-DAMP 同结构控制，三个种子分别新增 1、5、5 个危险检出，没有丢掉该控制已经检出的危险；每次新增 1 个 hard-safe 候选，同时取消 4 个控制的 hard-safe 候选，净少 3 个。这些配对事件保存在 `mechanism_disagreements.csv`。它展示了具体的筛选变化，仍是当前内部数据的观察，不能推广成所有未来场景都如此。

这支持“DAMP 构造作为整体，在保守的几何候选筛选框架内具有增量信息”。它不证明每个 stress、damage、persistence、recovery 项都不可缺少，也不证明免疫术语本身优于其他解释。已有单项消融中，有些删项影响很小，仍应如实报告。

## 4. 排序和提前量能否成为更强证据

在全部 249 个 TTC-negative 外部事件中，用相同长度的审阅队列比较。以下是 5% 版本的冻结分数，逐种子计算后平均的危险检出数；边界并列分数按并列组内随机排序的期望计数处理。

| 方式 | 审阅前 10 个 | 审阅前 25 个 | 审阅前 50 个 |
|---|---:|---:|---:|
""" + "\n".join(ranked) + """

中等短队列中有改善：前 25 个事件内，当前 RIR 平均发现 15.67 个危险，单帧 CPA 为 12 个；但前 10 个没有收益，前 50 个 CPA 反而找到更多。前 25 个的名义 p≈0.042，经本轮比较校正后 p≈0.422。因此可以作为描述性的 triage 证据，但不是全面排序优势的显著证明。

5% 版本的 AP 为 0.525 vs CPA 的 0.510；标准化 pAUC@10% FPR 为 0.710 vs 0.660。两者相对 CPA 的置信区间都跨零。10% 版本也类似。不能把 partial AUC 的局部优势单独挑出来当确定性结论。

时效改用“全部 33 个 hidden danger 中，至少提前 0.5 秒/1 秒检出的比例”，避免只在各自已检出的不同子集上比较平均 lead time。10% 预算下，至少提前 0.5 秒检出的比例为当前 RIR 29.29%、单帧 CPA 24.24%；经配对不确定性和多重比较考虑后仍不确定。提前 1 秒也不显示普遍优势。

这里的排序表按三种子平均计算，与正文既有一次固定排名中的 16/33 不是同一聚合口径，不应直接替换。

## 5. 改进原型的意义与局限

75% CPA / 25% DAMP 以较弱的 DAMP 惩罚保留中等学习证据但几何较强的目标；保留原有确认和旁路。10% 预算下三个种子均恢复 20 个危险，新增 hard-safe 候选分别为 18、18、23 个。候选率的三种子均值为 9.10%，其中一次为 10.65%，所以不能声称每次外部测试都被严格限制在 10% 以内。

相比当前版，hidden recall 增加 3.03 pp，95% 配对区间约 [-1.01,8.08]，描述性 p≈0.312；候选率增加 1.08 pp。三次种子的恢复数一致并不意味着新家族上的不确定性很小。

在 5% 预算下，该原型恢复率更高，但实际新增候选率达到 5.71%，明显高于当前的 3.24%。其代价增加经比较校正仍有统计支持，不能将召回增益描述成免费提升。

而且增加 CPA 权重带来的改进本身不是“DAMP 更有效”的证据。DAMP 的证据应来自上一节的机制控制。条件校准、联合尾概率及更宽松的时间汇总没有找到稳定超过当前版的结果；全部保留在报告中，不继续根据这 33 个危险样本微调分层或参数。

## 6. 建议的论文处理

- 保留单帧 CPA，并将带相同时间规则的 CPA、移除真实 DAMP、原始预测误差替代三个控制一起展示。这能区分“整体性能对比”和“机制增量价值”。
- 将最有支持的结论放在原先就已声明的 5% 保守操作点：DAMP 帮助几何候选筛选；10% 操作点作为审阅覆盖率与候选负担的权衡。
- 60.61% 原型作为新的探索性 sensitivity，先冻结后在新家族上确认，再考虑升级主方法；不直接替换摘要数字。
- 如继续投入模型改进，优先让预测器学习安全的横向/CPA 演化、观测不确定性和参与者几何，而不是在既有 33 个难例上继续搜索门控组合。新增感知特征必须同样提供给对照方法，不能用 GT 危险目标身份辅助本方法。

可供下一次论文修改参考的英文表述（不是已插入正文的文本）：

> At the prespecified conservative operating point, paired mechanism controls support an incremental contribution from DAMP-inspired evidence within the geometry-supported recognition pipeline. Replacing the learned evidence with a constant while retaining the same fusion transform, temporal aggregation, and severe-CPA bypass reduces hidden-danger recall from 44.44% to 33.33%, with an additional hard-safe candidate rate of 4.63% versus 3.24% for RIR. The paired recall difference is 11.11 percentage points (95% family-bootstrap interval: 3.03–20.20; Holm-adjusted descriptive p=0.036). These internal mechanistic results do not establish general superiority over peak-based CPA, for which the performance difference remains uncertain.

若正文使用新的 bootstrap 数字，应同时说明探索性性质、比较 family、三种子聚合口径和区间含义。

## 文件与复现

- `protocol.json` / `completion.json`：实验定义、源文件哈希和复现状态。
- `events.csv` / `thresholds.csv` / `seed_metrics.csv`：全部十种方法的事件输出、校准和指标。
- `fixed_review_ranking.csv`：所有固定队列长度结果，包括无提升和 CPA 更好的区域。
- `statistics_protocol.json` / `paired_statistics.csv`：完整配对区间、原始及 Holm 校正 p 值。
- `damp_evidence_audit.png` / `.svg`：供审阅的图，不是已验证的新主图。

```sh
XDG_CACHE_HOME=/private/tmp/codex-cache MPLCONFIGDIR=/private/tmp/mpl /Users/finn/anaconda3/envs/immune_driving/bin/python -m experiments.ego_immune_recognition.riskbench_metadata.damp_incremental_value_audit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.ego_immune_recognition.riskbench_metadata.damp_incremental_value_statistics
python3 -m experiments.ego_immune_recognition.riskbench_metadata.damp_incremental_value_report
```
"""
    (OUT / "DAMP_INCREMENTAL_VALUE_REPORT.md").write_text(report, encoding="utf-8")
    print(OUT / "DAMP_INCREMENTAL_VALUE_REPORT.md")


if __name__ == "__main__":
    main()
