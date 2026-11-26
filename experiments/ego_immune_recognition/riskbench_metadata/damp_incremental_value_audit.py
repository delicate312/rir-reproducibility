"""Frozen-head experiments testing whether learned DAMP adds value beyond CPA.

All definitions are fixed before this script scores outer data. This is
exploratory development on previously examined families, not confirmation.
Fit references and each threshold only on hard-safe validation events. No
winner is selected. Both existing 5% and 10% budgets and every row are saved.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np  # Import before torch in the immune_driving environment.
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import damp_decision_necessity_ablation as d

ROOT = d.PROJECT_ROOT
OUT = ROOT / "results/ego_immune_riskbench_damp_incremental_value_20260906"
BUDGETS = {
    .05: ("ego_immune_riskbench_damp_hard_safe_grid_v1", "ego_immune_riskbench_damp_necessity_v2_nested"),
    .10: ("ego_immune_riskbench_damp_hard_safe_grid_review10_v1", "ego_immune_riskbench_damp_necessity_review10_v1"),
}
METHODS = {
    "current": "sqrt(CPA * DAMP), consecutive mean of 3, severe CPA bypass",
    "cpa_peak": "CPA only, frame maximum (published ablation comparator)",
    "cpa_temporal": "CPA only, consecutive mean of 3, severe bypass",
    "constant_damp_control": "Replace DAMP percentile by 1; same sqrt, mean of 3 and bypass",
    "conditional_median_control": "Replace DAMP by validation median within CPA stratum; same rules",
    "raw_residual_hybrid": "Replace DAMP by causal forecast-residual percentile; same rules",
    "cpa_weighted": "CPA**0.75 * DAMP**0.25; same temporal rules and bypass",
    "conditional_damp": "DAMP percentile calibrated within CPA strata; same original sqrt and rules",
    "joint_tail": "One minus smoothed validation upper-orthant tail probability; same temporal rules and bypass",
    "two_of_five": "Original sqrt fusion; mean of highest 2 same-actor scores in last 5 positions; severe bypass",
}
CPA_BINS = np.asarray([0., .80, .95, .99, 1.000001])
MIN_STRATUM = 100


def upper_tail_score(reference, queries):
    """Exact 2-D upper-orthant counts with +1 smoothing, including ties.

    Scores are functions of the fixed reference and one query only; batched
    sorting changes runtime, not causal information available to a query.
    """
    reference = np.asarray(reference, float).reshape(-1, 2)
    queries = np.asarray(queries, float).reshape(-1, 2)
    if not len(reference):
        raise ValueError("Empty joint reference")
    ys = np.unique(reference[:, 1])
    tree = np.zeros(len(ys) + 1, dtype=int)

    def add(i):
        i += 1
        while i < len(tree):
            tree[i] += 1
            i += i & -i

    def prefix(n):
        total = 0
        while n:
            total += tree[n]
            n -= n & -n
        return total

    order = np.argsort(-reference[:, 0], kind="stable")
    result = np.empty(len(queries))
    inserted = 0
    for qi in np.argsort(-queries[:, 0], kind="stable"):
        x, y = queries[qi]
        while inserted < len(reference) and reference[order[inserted], 0] >= x:
            add(int(np.searchsorted(ys, reference[order[inserted], 1])))
            inserted += 1
        count = inserted - prefix(int(np.searchsorted(ys, y, side="left")))
        result[qi] = 1. - (count + 1.) / (len(reference) + 1.)
    return result


def conditional_values(reference, queries, *, median=False):
    output = np.zeros(len(queries))
    for lo, hi in zip(CPA_BINS[:-1], CPA_BINS[1:]):
        selected = (queries[:, 0] >= lo) & (queries[:, 0] < hi)
        local = reference[(reference[:, 0] >= lo) & (reference[:, 0] < hi), 1]
        if len(local) < MIN_STRATUM:
            local = reference[:, 1]
        if median:
            output[selected] = np.median(local)
        else:
            output[selected] = np.searchsorted(np.sort(local), queries[selected, 1], side="right") / len(local)
    return output


def temporal_scores(records, fused, cpa, *, support=3, window=None, bypass=True):
    """Causal same-actor pooling. records: (event UID, position, target ID)."""
    histories = defaultdict(list)
    scores = defaultdict(float)
    for (uid, position, target), value, cp in zip(records, fused, cpa):
        history = histories[(uid, target)]
        if window is None:
            if history and position != history[-1][0] + 1:
                history.clear()
            history.append((position, float(value)))
            history[:] = history[-support:]
        else:
            history[:] = [item for item in history if item[0] >= position - window + 1]
            history.append((position, float(value)))
        score = float(np.mean(sorted([v for _, v in history], reverse=True)[:support])) if len(history) >= support else 0.
        if bypass and cp >= d.SEVERE_CPA_PERCENTILE:
            score = max(score, float(cp))
        scores[(uid, position)] = max(scores[(uid, position)], score)
    return scores


def pair_arrays(events, components, geometry, activation, damp_ref, cpa_ref, raw_ref, *, usable_only):
    records, values = [], []
    for uid, frames in events.items():
        for frame in d._usable_frames(frames) if usable_only else frames:
            pos = int(frame["frame_position"])
            comp = components[uid].get(pos, {})
            for target, geo in geometry.get((uid, pos), {}).items():
                item = comp.get(str(target))
                if item is None:
                    continue
                records.append((uid, pos, str(target)))
                values.append((d._percentile(cpa_ref, geo["raw"]),
                               d._percentile(damp_ref, activation(item)),
                               d._percentile(raw_ref, item["causal_raw"])))
    return records, np.asarray(values, float).reshape(-1, 3)


def all_head_scores(records, values, joint_reference):
    c, a, raw = values.T
    base = np.sqrt(c * a)
    conditional = conditional_values(joint_reference, values[:, :2])
    median = conditional_values(joint_reference, values[:, :2], median=True)
    fused = {
        "current": base, "cpa_peak": c, "cpa_temporal": c,
        "constant_damp_control": np.sqrt(c),
        "conditional_median_control": np.sqrt(c * median),
        "raw_residual_hybrid": np.sqrt(c * raw),
        "cpa_weighted": c**.75 * a**.25,
        "conditional_damp": np.sqrt(c * conditional),
        "joint_tail": upper_tail_score(joint_reference, values[:, :2]),
        "two_of_five": base,
    }
    return {
        name: temporal_scores(records, score, c,
                              support=1 if name == "cpa_peak" else 2 if name == "two_of_five" else 3,
                              window=5 if name == "two_of_five" else None,
                              bypass=name != "cpa_peak")
        for name, score in fused.items()
    }


def event_outputs(events, frame_scores):
    output = {}
    for uid, frames in events.items():
        usable = d._usable_frames(frames)
        trace = [(frame, frame_scores.get((uid, int(frame["frame_position"])), 0.)) for frame in usable]
        output[uid] = (d.event_score(trace), trace)
    return output


def attach_official_cpa(heads, events, geometry, cpa_reference):
    # CPA can score a newly observed actor even when no forecast residual is
    # available. Preserve that access exactly as in the formal comparator.
    for name in ("cpa_peak", "cpa_temporal"):
        heads[name] = {
            (uid, int(frame["frame_position"])): score
            for uid, frames in events.items()
            for frame, score in d._cpa_trace(frames, geometry, cpa_reference,
                confirmation=name == "cpa_temporal", severe_bypass=name == "cpa_temporal")
        }


def main():
    OUT.mkdir(exist_ok=True)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), methods=METHODS,
        cpa_bins=CPA_BINS.tolist(), min_stratum=MIN_STRATUM, budgets=list(BUDGETS),
        seeds=list(d.DEFAULT_SEEDS), source_sha256=d.sha256_file(Path(__file__)),
        claim_status="exploratory_no_winner_selection_no_confirmatory_claim",
        fits="validation_normal only; existing budget-specific DAMP coefficients; independent event thresholds",
        test_labels="metrics only; no parameter or threshold selection", source_hashes={})
    for budget, (selection, formal) in BUDGETS.items():
        protocol["source_hashes"][str(budget)] = {
            "selection": d.sha256_file(ROOT / "results" / selection / "selected_parameters.csv"),
            "formal": d.sha256_file(ROOT / "results" / formal / "outer_event_predictions.csv")}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows, threshold_rows = [], []
    formal_tables = {budget: pd.read_csv(ROOT / "results" / formal / "outer_event_predictions.csv").set_index(["seed", "fold", "event_uid", "method"])
                     for budget, (_, formal) in BUDGETS.items()}
    for fold in range(5):
        print(f"Fold {fold}: loading causal geometry", flush=True)
        index = d.DEFAULT_INDEX_ROOT / f"fold_{fold:02d}/riskbench_nested_fold_index.json"
        observations = d._observations_by_frame(index, splits=("validation_normal", "test_normal", "test_positive"))
        geometry = d._geometry_from_observations(observations, horizon_seconds=2., clearance_m=3.5)
        payload = json.loads(index.read_text())
        metadata = {f"riskbench:{r['category']}:{r['scenario_id']}": r for sp in ("validation_normal", "test_normal", "test_positive") for r in payload["splits"][sp]}
        for seed in d.DEFAULT_SEEDS:
            frames = d._load_frames(d.DEFAULT_SOURCE_ROOT, seed, fold)
            events = {sp: d._event_frames(frames[sp]) for sp in ("validation_normal", "test_normal", "test_positive")}
            val = events["validation_normal"]
            physical = d._physical_reference(val)
            cpa_ref = d._cpa_reference(val, geometry)
            ttc = {sp: {uid: d.event_score(d._physical_trace(fs, physical)) for uid, fs in ev.items()} for sp, ev in events.items()}
            val_negative = sorted(uid for uid, score in ttc["validation_normal"].items() if score < .95)
            for budget, (selection_root, _) in BUDGETS.items():
                candidate = d._selected_candidate(ROOT / "results" / selection_root, seed=seed, fold=fold)
                # Only validation data are used in all fitting operations below.
                val_components = d._merged_components(val, d._recompute_components(val, persistence_decay=candidate.persistence_decay, recovery_warmup=candidate.recovery_warmup))
                calibration = d._calibration(val, val_components)
                activation = d._activation_fn("rir_nested_selected", calibration, candidate)
                damp_ref = d._activation_reference(val_components, activation)
                raw_ref = np.sort([v["causal_raw"] for positions in val_components.values() for targets in positions.values() for v in targets.values()])
                _, ref_pairs = pair_arrays(val, val_components, geometry, activation, damp_ref, cpa_ref, raw_ref, usable_only=False)
                joint_reference = ref_pairs[:, :2]
                records, values = pair_arrays(val, val_components, geometry, activation, damp_ref, cpa_ref, raw_ref, usable_only=True)
                validation_heads = all_head_scores(records, values, joint_reference)
                attach_official_cpa(validation_heads, val, geometry, cpa_ref)
                thresholds = {}
                for name, frame_scores in validation_heads.items():
                    summaries = event_outputs(val, frame_scores)
                    threshold, val_rate = d.threshold_at_event_fpr([summaries[uid][0] for uid in val_negative], budget)
                    assert val_rate <= budget + 1e-12
                    thresholds[name] = threshold
                    threshold_rows.append(dict(fold=fold, seed=seed, budget=budget, method=name, threshold=threshold, val_rate=val_rate, val_negative_n=len(val_negative)))
                # Persist the fitted heads' thresholds before any outer scoring.
                pd.DataFrame(threshold_rows).to_csv(OUT / "thresholds.csv", index=False)
                for split in ("test_normal", "test_positive"):
                    ev = events[split]
                    components = d._merged_components(ev, d._recompute_components(ev, persistence_decay=candidate.persistence_decay, recovery_warmup=candidate.recovery_warmup))
                    records, values = pair_arrays(ev, components, geometry, activation, damp_ref, cpa_ref, raw_ref, usable_only=True)
                    heads = all_head_scores(records, values, joint_reference)
                    attach_official_cpa(heads, ev, geometry, cpa_ref)
                    for name, frame_scores in heads.items():
                        for uid, (score, trace) in event_outputs(ev, frame_scores).items():
                            threshold = thresholds[name]
                            hit, lead = d.event_alert(trace, threshold)
                            ttc_hit = int(ttc[split][uid] >= .95)
                            if name in {"current", "cpa_peak", "cpa_temporal"}:
                                official = {"current": "rir_nested_selected", "cpa_peak": "cpa_only", "cpa_temporal": "cpa_temporal_matched"}[name]
                                expected = formal_tables[budget].loc[(seed, fold, uid, official)]
                                np.testing.assert_allclose([score, threshold, ttc[split][uid]], expected[["method_score", "threshold", "ttc_score"]].to_numpy(float), atol=1e-10, rtol=0)
                                assert int(hit or ttc_hit) == expected.final_detected
                            rows.append(dict(seed=seed, fold=fold, budget=budget, method=name, split=split, event_uid=uid,
                                family=metadata[uid]["scenario_group_id"], target_type=metadata[uid]["target_type_name"],
                                score=score, threshold=threshold, ttc_detected=ttc_hit, relation_detected=int(hit),
                                detected=int(hit or ttc_hit), lead_seconds=lead,
                                hit_by_0p5s=int(any(s >= threshold and int(f.get("eval_frames_to_anchor", f["frames_to_anchor"])) >= 10 for f, s in trace)),
                                hit_by_1s=int(any(s >= threshold and int(f.get("eval_frames_to_anchor", f["frames_to_anchor"])) >= 20 for f, s in trace))))
                print(f"fold {fold}, seed {seed}, budget {budget}: 3 formal controls reproduced; all heads scored", flush=True)
                pd.DataFrame(rows).to_csv(OUT / "events.csv", index=False)
    events_df = pd.DataFrame(rows)
    metrics = []
    ranking = []
    for (budget, method, seed), subset in events_df.groupby(["budget", "method", "seed"]):
        pos = subset[subset.split.eq("test_positive")]
        blind = subset[subset.ttc_detected.eq(0)]
        hidden = blind[blind.split.eq("test_positive")]
        safe = blind[blind.split.eq("test_normal")]
        y = blind.split.eq("test_positive").to_numpy(int)
        metrics.append(dict(budget=budget, method=method, seed=seed,
            overall_recall=100*pos.detected.mean(), hidden_recall=100*hidden.detected.mean(), added_rate=100*safe.detected.mean(),
            hidden_AP=average_precision_score(y, blind.score), hidden_AUROC=roc_auc_score(y, blind.score),
            hidden_pAUC10=roc_auc_score(y, blind.score, max_fpr=.10),
            recall_by_0p5s=100*hidden.hit_by_0p5s.mean(), recall_by_1s=100*hidden.hit_by_1s.mean(),
            rescued=int(hidden.detected.sum()), added=int(safe.detected.sum())))
        # Fixed ranking budgets, never promoted to validation-calibrated thresholds.
        ordered = blind.sort_values(["score", "event_uid"], ascending=[False, True])
        for k in (10, 25, 50):
            cutoff = float(ordered.iloc[k-1].score)
            strict = ordered[ordered.score > cutoff]
            tied = ordered[ordered.score.eq(cutoff)]
            expected_hits = strict.split.eq("test_positive").sum() + (k-len(strict))*tied.split.eq("test_positive").mean()
            ranking.append(dict(budget=budget, method=method, seed=seed, k=k, expected_hits=expected_hits,
                                recall=100*expected_hits/33, precision=100*expected_hits/k, ties_at_cutoff=len(tied)))
    pd.DataFrame(metrics).to_csv(OUT / "seed_metrics.csv", index=False)
    pd.DataFrame(ranking).to_csv(OUT / "fixed_review_ranking.csv", index=False)
    summary = pd.DataFrame(metrics).groupby(["budget", "method"])[["overall_recall", "hidden_recall", "added_rate", "hidden_AP", "hidden_AUROC", "hidden_pAUC10", "recall_by_0p5s", "recall_by_1s"]].agg(["mean", "std"])
    summary.to_csv(OUT / "summary.csv")
    print(summary.to_string(), flush=True)
    protocol.update(completed_utc=datetime.now(timezone.utc).isoformat(), formal_reproduction="current, CPA peak and CPA temporal; both budgets; all 15 runs passed", event_rows=len(rows))
    (OUT / "completion.json").write_text(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
