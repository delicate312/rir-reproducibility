"""Paired family uncertainty, including all planned controls and prototypes.

Stratified family bootstrap keeps the hidden population at 33 danger and
216 hard-safe families. The same weights are used for every method and
seed; metrics are averaged over seeds, not computed on an ensemble score.
All p values remain descriptive for previously inspected internal data.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/ego_immune_riskbench_damp_incremental_value_20260906"
REPLICATES = 10000
SEEDS = (20260812, 20260813, 20260814)
CONTROLS = ("cpa_peak", "cpa_temporal", "constant_damp_control", "conditional_median_control", "raw_residual_hybrid")
PROTOTYPES = ("cpa_weighted", "conditional_damp", "joint_tail", "two_of_five")


def weighted_curve_metrics(y, scores, weights):
    """Vectorized exact AP/ROC/pAUC and randomized-within-tie top-K metrics."""
    y = np.asarray(y, int)
    scores = np.asarray(scores, float)
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(scores)-1]
    w = weights[:, order]
    tp = np.cumsum(w * y[order], axis=1)[:, ends]
    fp = np.cumsum(w * (1-y[order]), axis=1)[:, ends]
    zero = np.zeros((len(weights), 1))
    ptp, pfp = np.c_[zero, tp[:, :-1]], np.c_[zero, fp[:, :-1]]
    dp, dn = tp - ptp, fp - pfp
    npos, nneg = tp[:, -1:], fp[:, -1:]
    precision = np.divide(tp, tp+fp, out=np.zeros_like(tp, float), where=(tp+fp)>0)
    ap = np.sum(dp/npos * precision, axis=1)
    tpr, fpr = tp/npos, fp/nneg
    tpr0, fpr0 = ptp/npos, pfp/nneg
    dx = fpr-fpr0
    auc = np.sum(dx*(tpr+tpr0)/2, axis=1)
    clipped_dx = np.clip(np.minimum(fpr, .1)-fpr0, 0, None)
    portion = np.divide(clipped_dx, dx, out=np.zeros_like(dx), where=dx>0)
    end_tpr = tpr0 + (tpr-tpr0)*portion
    area = np.sum(clipped_dx*(tpr0+end_tpr)/2, axis=1)
    pauc = .5*(1+(area-.5*.1**2)/(.1-.5*.1**2))
    result = dict(AP=ap, AUROC=auc, pAUC10=pauc)
    for k in (10, 25, 50):
        fraction = np.clip(np.divide(k-(ptp+pfp), dp+dn,
            out=np.zeros_like(dp, float), where=(dp+dn)>0), 0, 1)
        result[f"top{k}_hits"] = np.sum(dp*fraction, axis=1)
    return result


def holm(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    result = np.empty(len(pvalues))
    result[order] = np.minimum(1, np.maximum.accumulate(pvalues[order]*(len(pvalues)-np.arange(len(pvalues)))))
    return result


def main():
    # Write the analysis family definitions before computing uncertainty.
    plan = dict(replicates=REPLICATES, random_seed=20260906,
        paired_unit="scenario family; same resampling weights across seeds and methods",
        controls=CONTROLS, prototypes=PROTOTYPES,
        multiplicity="Holm across all contrasts and both budgets within evidence family and metric",
        evidence_families={"mechanism": "current versus every control", "prototype": "every new head versus current"},
        status="posthoc internal analysis; intervals and tests are descriptive")
    (OUT / "statistics_protocol.json").write_text(json.dumps(plan, indent=2))
    events = pd.read_csv(OUT / "events.csv")
    events = events[events.ttc_detected.eq(0)]
    unique = events[["family", "split"]].drop_duplicates().sort_values("family")
    y = unique.split.eq("test_positive").to_numpy(int)
    assert y.sum() == 33 and (1-y).sum() == 216 and len(unique) == 249
    rng = np.random.default_rng(20260906)
    weights = np.ones((REPLICATES+1, len(y)), dtype=np.int16)
    for klass, n in [(1, 33), (0, 216)]:
        weights[1:, y==klass] = rng.multinomial(n, np.full(n, 1/n), size=REPLICATES)
    distributions = {}
    for (budget, method), subset in events.groupby(["budget", "method"]):
        seed_values = []
        for seed in SEEDS:
            frame = subset[subset.seed.eq(seed)].set_index("family").loc[unique.family]
            values = weighted_curve_metrics(y, frame.score.to_numpy(), weights)
            values["hidden_recall"] = weights[:, y==1] @ frame.loc[y==1, "detected"].to_numpy(float) / 33 * 100
            values["added_rate"] = weights[:, y==0] @ frame.loc[y==0, "detected"].to_numpy(float) / 216 * 100
            for col in ("hit_by_0p5s", "hit_by_1s"):
                values[col] = weights[:, y==1] @ frame.loc[y==1, col].to_numpy(float) / 33 * 100
            seed_values.append(values)
        distributions[(budget, method)] = {name: np.mean([v[name] for v in seed_values], axis=0) for name in seed_values[0]}
        print(f"bootstrap curves: {budget} {method}", flush=True)
    # Validate the non-resampled metric row against the independently computed
    # sklearn outputs from the scoring script.
    official = pd.read_csv(OUT / "seed_metrics.csv").groupby(["budget", "method"]).mean(numeric_only=True)
    for key, values in distributions.items():
        for metric, field in [("AP", "hidden_AP"), ("AUROC", "hidden_AUROC"), ("pAUC10", "hidden_pAUC10"), ("hidden_recall", "hidden_recall"), ("added_rate", "added_rate")]:
            np.testing.assert_allclose(values[metric][0], official.loc[key, field], atol=1e-12)
    rows = []
    for budget in (.05, .10):
        contrasts = [("mechanism", "current", c) for c in CONTROLS] + [("prototype", p, "current") for p in PROTOTYPES]
        for family, method, reference in contrasts:
            for metric, values in distributions[(budget, method)].items():
                delta = values-distributions[(budget, reference)][metric]
                draws = delta[1:]
                p = min(1., 2*min((1+(draws<=0).sum())/(REPLICATES+1), (1+(draws>=0).sum())/(REPLICATES+1)))
                rows.append(dict(evidence_family=family,budget=budget,method=method,reference=reference,metric=metric,
                    method_value=values[0], reference_value=distributions[(budget, reference)][metric][0],
                    difference=delta[0], ci95_low=np.quantile(draws,.025),ci95_high=np.quantile(draws,.975),p_unadjusted=p))
    table = pd.DataFrame(rows)
    for _, indices in table.groupby(["evidence_family", "metric"]).groups.items():
        table.loc[indices, "p_holm"] = holm(table.loc[indices, "p_unadjusted"])
    table.to_csv(OUT / "paired_statistics.csv", index=False)
    print(table[table.metric.isin(["hidden_recall", "pAUC10", "top25_hits"])].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
