"""Frozen exploratory test of history beyond present CPA, with release diagnostics.

No model training or positive-label parameter selection. The already examined
outer families are development evidence, not a new independent test set.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np  # Must precede torch imports in this environment.
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import damp_decision_necessity_ablation as d
from .damp_incremental_value_audit import BUDGETS

OUT = d.PROJECT_ROOT / "results/ego_immune_riskbench_history_geometry_20260906"
LANDMARKS = (20, 10, 5)  # 1.0, 0.5, 0.25 seconds BEFORE the official anchor.
METHODS = ("cpa_peak", "cpa_temporal", "cpa_ema", "cpa_trend", "cpa_cusum", "rir", "rir_no_recovery")
SNAPSHOT_METHODS = METHODS + ("damp", "raw_residual")


def lead(frame):
    return int(frame.get("eval_frames_to_anchor", frame["frames_to_anchor"]))


def history_controls(positions, cpa):
    """Causal CPA history controls; reset on observation gaps, no fitted weights."""
    result = []
    history = []
    ema = cusum = 0.
    previous = None
    for pos, c in zip(positions, cpa):
        if previous is None or pos != previous + 1:
            history = []
            ema, cusum = float(c), 0.
        else:
            ema = .8 * ema + .2 * c
        history.append(float(c))
        history = history[-10:]
        cusum = max(0., .9 * cusum + c - .95)
        temporal = np.mean(history[-3:]) if len(history) >= 3 else 0.
        if c >= d.SEVERE_CPA_PERCENTILE:
            temporal = max(temporal, c)
        if len(history) >= 3:
            x = np.arange(len(history), dtype=float)
            x -= x.mean()
            slope = np.dot(x, history) / np.dot(x, x)
            trend = max(0., c + 10 * slope)  # Extrapolate the CPA score by 0.5 s.
        else:
            trend = float(c)
        result.append((float(c), float(temporal), float(ema), float(trend), float(cusum)))
        previous = pos
    return result


def pair_and_scene_scores(events, components, geometry, cpa_ref, activation, damp_ref,
                          activation_nr, damp_ref_nr, raw_ref):
    pair, scene = {}, {}
    for uid, frames in events.items():
        selected = sorted((f for f in frames if -20 <= lead(f) <= 30), key=lambda f: f["frame_position"])
        by_actor = defaultdict(list)
        for f in selected:
            pos = int(f["frame_position"])
            scene[(uid, pos)] = dict.fromkeys(METHODS, 0.)
            for target, geo in geometry.get((uid, pos), {}).items():
                by_actor[str(target)].append((pos, geo))
        for target, observations in by_actor.items():
            positions = [p for p, _ in observations]
            cs = [d._percentile(cpa_ref, geo["raw"]) for _, geo in observations]
            controls = history_controls(positions, cs)
            joint_history = []
            for (pos, geo), cp, ctrl in zip(observations, cs, controls):
                scores = dict(zip(METHODS[:5], ctrl))
                values = components[uid].get(pos, {}).get(target)
                scores.update(rir=0., rir_no_recovery=0., damp=np.nan, raw_residual=np.nan)
                if values is None:
                    joint_history = []
                else:
                    dp = d._percentile(damp_ref, activation(values))
                    nr = d._percentile(damp_ref_nr, activation_nr(values))
                    scores.update(damp=dp, raw_residual=d._percentile(raw_ref, values["causal_raw"]))
                    if joint_history and pos != joint_history[-1][0] + 1:
                        joint_history = []
                    joint_history.append((pos, np.sqrt(cp * dp), np.sqrt(cp * nr)))
                    joint_history = joint_history[-3:]
                    if len(joint_history) == 3:
                        scores["rir"], scores["rir_no_recovery"] = np.mean(np.array(joint_history)[:, 1:], axis=0)
                    if cp >= d.SEVERE_CPA_PERCENTILE:
                        for name in ("rir", "rir_no_recovery"):
                            scores[name] = max(scores[name], cp)
                pair[(uid, pos, target)] = scores
                for name in METHODS:
                    scene[(uid, pos)][name] = max(scene[(uid, pos)][name], scores[name])
    return pair, scene


def match_snapshots(positive, negative, mode):
    """Maximum-cardinality, minimum-distance, within-fold one-to-one matching.

    No model score enters matching. Both actors must have a future CPA with
    nonzero raw score and an available causal residual. Scores rank EVENTS,
    not GT risk actors: target selection uses present geometry alone.
    """
    if len(positive) == 0 or len(negative) == 0:
        return []
    a, b = positive, negative
    dc = np.abs(a.cpa_raw.to_numpy()[:, None] - b.cpa_raw.to_numpy()[None, :])
    dp = np.sqrt(sum((a[k].to_numpy()[:, None] - b[k].to_numpy()[None, :])**2 for k in ("px", "py")))
    dv = np.sqrt(sum((a[k].to_numpy()[:, None] - b[k].to_numpy()[None, :])**2 for k in ("vx", "vy")))
    dt = np.abs(a.tcpa.to_numpy()[:, None] - b.tcpa.to_numpy()[None, :])
    dd = np.abs(a.dcpa.to_numpy()[:, None] - b.dcpa.to_numpy()[None, :])
    if mode == "geometry":
        valid = (dc <= .05) & (dp <= 5.) & (dv <= 2.) & (dt <= .25) & (dd <= 1.)
        distance = dc/.05 + dp/5 + dv/2 + dt/.25 + dd
    elif mode == "cpa_only":
        valid = dc <= .02
        distance = dc/.02
    else:
        raise ValueError(mode)
    # Dummy assignments permit unmatched cases; a huge fixed penalty makes
    # cardinality dominate distance, including for the full 420-event corpus.
    cost = np.full((len(a), len(b) + len(a)), 1e6)
    cost[:, :len(b)] = np.where(valid, distance, 1e9)
    ii, jj = linear_sum_assignment(cost)
    return [(int(i), int(j), float(dc[i, j]), float(dp[i, j]), float(dv[i, j]), float(dt[i, j]), float(dd[i, j]))
            for i, j in zip(ii, jj) if j < len(b) and valid[i, j]]


def release_diagnostic(frames, scores, threshold, behavior_end):
    """Behavior-end proxy only; no assertion that behavior end == danger end."""
    lookup = {int(f["source_frame_index"]): (f, s) for f, s in zip(frames, scores)}
    end = int(behavior_end)
    complete = all(t in lookup for t in range(end - 9, end + 21))
    out = dict(complete=int(complete), pre_alert=0, post_occupancy=np.nan,
               release_confirm_s=np.nan, release_censored=1, pre_occupancy=np.nan)
    if not complete:
        return out
    pre = np.array([lookup[t][1] >= threshold for t in range(end-9, end+1)])
    post = np.array([lookup[t][1] >= threshold for t in range(end+1, end+21)])
    out.update(pre_alert=int(pre.any()), pre_occupancy=float(pre.mean()), post_occupancy=float(post.mean()))
    # End of first 5 consecutive unalerted observations, evaluated online.
    for j in range(4, 20):
        if not post[j-4:j+1].any():
            out.update(release_confirm_s=(j+1)/20, release_censored=0)
            break
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status="exploratory_on_previously_examined_families; not independent confirmation",
        source_sha256=d.sha256_file(Path(__file__)), seeds=list(d.DEFAULT_SEEDS), budgets=list(BUDGETS),
        methods=list(METHODS), snapshot_methods=list(SNAPSHOT_METHODS), landmarks_frames=LANDMARKS,
        matching="within outer fold, unique pairs at each landmark; primary full geometry, secondary CPA only; all events primary, TTC-negative secondary",
        geometry_calipers=dict(cpa=.05, position_m=5, velocity_mps=2, tcpa_s=.25, dcpa_m=1), cpa_only_caliper=.02,
        eligibility="selected current max-CPA actor (tie: nearest then actor ID); raw CPA > 0 and available residual; no GT actor selection",
        controls="CPA EMA decay .8; 10-frame OLS CPA-score trend extrapolated 10 frames; leaky CUSUM max(0,.9*state+CPA_percentile-.95); gaps reset",
        history_scope="score controls start at anchor-30; cached learned components retain original causal warmup; no future frames used in scores",
        thresholds="independently calibrated on TTC-negative hard-safe validation events over original pre-anchor 30-frame window",
        release="secondary descriptive behavior-end proxy; requires last 10 pre and all 20 post observations; paired common pre-alert comparison; no true danger-offset labels",
        statistics="10000 paired bootstrap draws, seed metrics averaged first; matched pairs resampled within a landmark; event metrics resample families; pointwise intervals; Holm across reported contrasts within each analysis",
        primary_comparison="RIR versus each CPA control on all-danger recall by fixed lead at both budgets; matched DAMP versus each CPA control at each landmark; no winning horizon selected",
        dependencies={str(p): d.sha256_file(p) for p in [Path(d.__file__), Path(__file__).with_name("damp_hard_safe_grid_selection.py")]})
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        if old["source_sha256"] != protocol["source_sha256"]:
            raise RuntimeError("Source changed after protocol freeze; preserve old run before restarting")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2))
    snapshots, event_rows, threshold_rows, releases, availability = [], [], [], [], []
    formal_tables = {budget: pd.read_csv(d.PROJECT_ROOT / "results" / formal / "outer_event_predictions.csv").set_index(["seed", "fold", "event_uid", "method"])
                     for budget, (_, formal) in BUDGETS.items()}
    for fold in range(5):
        print(f"fold {fold}: loading geometry", flush=True)
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
                candidate = d._selected_candidate(d.PROJECT_ROOT / "results" / selection_root, seed=seed, fold=fold)
                comp = {sp: d._merged_components(ev, d._recompute_components(ev, persistence_decay=candidate.persistence_decay, recovery_warmup=candidate.recovery_warmup)) for sp, ev in events.items()}
                calibration = d._calibration(val, comp["validation_normal"])
                activation = d._activation_fn("rir_nested_selected", calibration, candidate)
                activation_nr = d._activation_fn("rir_nested_selected", calibration, replace(candidate, recovery_beta=0.))
                damp_ref = d._activation_reference(comp["validation_normal"], activation)
                damp_ref_nr = d._activation_reference(comp["validation_normal"], activation_nr)
                raw_ref = np.sort([v["causal_raw"] for positions in comp["validation_normal"].values() for targets in positions.values() for v in targets.values()])
                pair_val, scene_val = pair_and_scene_scores(val, comp["validation_normal"], geometry, cpa_ref, activation, damp_ref, activation_nr, damp_ref_nr, raw_ref)
                thresholds = {}
                for method in METHODS:
                    vals = [max((scene_val[(uid, int(f["frame_position"]))][method] for f in d._usable_frames(val[uid])), default=0.) for uid in val_negative]
                    th, rate = d.threshold_at_event_fpr(vals, budget)
                    assert rate <= budget + 1e-12
                    thresholds[method] = th
                    threshold_rows.append(dict(seed=seed, fold=fold, budget=budget, method=method, threshold=th, val_rate=rate, val_n=len(vals), recovery_beta=candidate.recovery_beta))
                pd.DataFrame(threshold_rows).to_csv(OUT / "thresholds.csv", index=False)
                for split in ("test_normal", "test_positive"):
                    ev = events[split]
                    pair, scene = pair_and_scene_scores(ev, comp[split], geometry, cpa_ref, activation, damp_ref, activation_nr, damp_ref_nr, raw_ref)
                    for uid, fs in ev.items():
                        common = dict(seed=seed, fold=fold, budget=budget, event_uid=uid, family=metadata[uid]["scenario_group_id"], positive=int(split == "test_positive"), ttc_detected=int(ttc[split][uid] >= .95))
                        usable = d._usable_frames(fs)
                        for method in METHODS:
                            trace = [(f, scene[(uid, int(f["frame_position"]))][method]) for f in usable]
                            score = d.event_score(trace)
                            th = thresholds[method]
                            if method in ("rir", "cpa_peak", "cpa_temporal"):
                                official = {"rir": "rir_nested_selected", "cpa_peak": "cpa_only", "cpa_temporal": "cpa_temporal_matched"}[method]
                                expected = formal_tables[budget].loc[(seed, fold, uid, official)]
                                np.testing.assert_allclose([score, th, ttc[split][uid]], expected[["method_score", "threshold", "ttc_score"]].to_numpy(float), atol=1e-10, rtol=0)
                            row = dict(common, method=method, score=score, threshold=th, hit=int(score >= th))
                            for horizon in (30, 20, 10, 5, 1):
                                row[f"hit_by_{horizon}"] = int(any(s >= th and lead(f) >= horizon for f, s in trace))
                            event_rows.append(row)
                            bw = metadata[uid].get("behavior_window")
                            if split == "test_normal" and bw:
                                allfs = sorted((f for f in fs if -20 <= lead(f) <= 30), key=lambda f: f["frame_position"])
                                diag = release_diagnostic(allfs, [scene[(uid, int(f["frame_position"]))][method] for f in allfs], th, bw[1])
                                releases.append(dict(common, method=method, behavior_end=bw[1], critical_frame=metadata[uid]["critical_frame"], **diag))
                        for landmark in LANDMARKS:
                            at = [f for f in fs if lead(f) == landmark]
                            avail = dict(common, landmark=landmark, available_frame=int(bool(at)), geometry_available=0, residual_available=0, nonzero_cpa=0)
                            if at:
                                pos = int(at[0]["frame_position"])
                                geos = geometry.get((uid, pos), {})
                                if geos:
                                    target = min(geos, key=lambda k: (-geos[k]["raw"], np.linalg.norm(observations[(uid, pos)][k][0]), str(k)))
                                    geo = geos[target]
                                    p, v = observations[(uid, pos)][target]
                                    values = pair[(uid, pos, str(target))]
                                    avail.update(geometry_available=1, residual_available=int(np.isfinite(values["damp"])), nonzero_cpa=int(geo["raw"] > 0))
                                    snapshots.append(dict(common, landmark=landmark, target=str(target), target_is_gt=int(str(target) in [str(k) for k in metadata[uid]["risk_actor_ids"]]),
                                        px=p[0], py=p[1], vx=v[0], vy=v[1], cpa_raw=geo["raw"], tcpa=geo["tcpa_seconds"], dcpa=geo["dcpa_m"], **values))
                            availability.append(avail)
                print(f"fold {fold}, seed {seed}, budget {budget}: original RIR and CPA reproduced; histories and post-anchor traces scored", flush=True)
        for name, rows in (("snapshots", snapshots), ("events", event_rows), ("release_proxy", releases), ("availability", availability)):
            pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    (OUT / "completion.json").write_text(json.dumps(dict(completed_utc=datetime.now(timezone.utc).isoformat(), source_sha256=d.sha256_file(Path(__file__)), formal_reproduction="all 15 folds/seeds, both budgets, RIR and two CPA controls", event_rows=len(event_rows), snapshot_rows=len(snapshots)), indent=2))


if __name__ == "__main__":
    main()
