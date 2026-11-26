from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.ego_immune_recognition.temporal_relation_train_eval import (
    _apply_training_safety_policy,
    _counterfactual_severity,
    _ewma_update,
    _frame_auc_views,
    _label_window_mask_from_frames,
    _observable_divergence,
    _paired_counterfactual_diagnostics,
    _roc_auc,
    _top_edge_hit_type,
    _three_pool_protocol_audit,
)


class EgoTemporalRelationTrainEvalTest(unittest.TestCase):
    def test_three_pool_audit_enforces_gradient_boundary(self) -> None:
        class Frame:
            def __init__(self, alert: int = 0, physical: int = 0) -> None:
                self.y_immune_alert = torch.tensor(alert)
                self.y_physical_danger = torch.tensor(physical)

        audit = _three_pool_protocol_audit(
            safe_core=[[Frame()]],
            hard_normal=[[Frame()]],
            positive_ood=[[Frame(alert=1)]],
        )

        self.assertEqual(audit["optimization_pool"], "safe_core_only")
        self.assertEqual(audit["hard_normal_gradient_steps"], 0)
        self.assertEqual(audit["positive_ood_gradient_steps"], 0)
        with self.assertRaisesRegex(ValueError, "safe-core"):
            _three_pool_protocol_audit(
                safe_core=[[Frame(alert=1)]],
                hard_normal=[[Frame()]],
                positive_ood=[[Frame(alert=1)]],
            )

    def test_safe_training_policy_splits_memory_at_risk_frames(self) -> None:
        sequence = ["safe-a", "safe-b", "risk", "safe-c", "safe-d"]

        def scores(frame: str) -> dict[str, float]:
            risk = frame == "risk"
            return {
                "inverse_ttc_ego_max": 0.3 if risk else 0.0,
                "closing_speed_ego_max": 4.0 if risk else 0.0,
            }

        with patch(
            "experiments.ego_immune_recognition.temporal_relation_train_eval."
            "_ego_rule_scores_for_graph",
            side_effect=scores,
        ):
            segments = _apply_training_safety_policy(
                [sequence],
                policy="kinematic_safe_segments",
                min_segment_frames=2,
            )

        self.assertEqual(segments, [["safe-a", "safe-b"], ["safe-c", "safe-d"]])

    def test_counterfactual_severity_uses_fixed_delay_bins(self) -> None:
        self.assertEqual(_counterfactual_severity(5), "mild")
        self.assertEqual(_counterfactual_severity(9), "medium")
        self.assertEqual(_counterfactual_severity(20), "strong")
        self.assertEqual(_counterfactual_severity(None), "unknown")

    def test_observable_divergence_respects_uncertainty_band(self) -> None:
        normal = {
            "ego_id": "ego",
            "nodes": {
                "ego": {
                    "position": np.array([0.0, 0.0]),
                    "velocity": np.array([0.0, 0.0]),
                    "confidence": 1.0,
                },
                "a": {
                    "position": np.array([10.0, 0.0]),
                    "velocity": np.array([0.0, 0.0]),
                    "confidence": 1.0,
                },
            },
            "edges": {
                ("ego", "a"): {
                    "distance": 10.0,
                    "closing_speed": 0.0,
                    "inverse_ttc": 0.0,
                    "confidence": 1.0,
                }
            },
        }
        counterfactual = {
            **normal,
            "nodes": {
                **normal["nodes"],
                "a": {
                    "position": np.array([10.8, 0.0]),
                    "velocity": np.array([0.5, 0.0]),
                    "confidence": 1.0,
                },
            },
            "edges": {
                ("ego", "a"): {
                    "distance": 9.2,
                    "closing_speed": 0.6,
                    "inverse_ttc": 0.1,
                    "confidence": 1.0,
                }
            },
        }

        result = _observable_divergence(normal, counterfactual)

        self.assertEqual(result["observable_divergence"], 1)
        self.assertEqual(result["position_observable"], 1)
        self.assertEqual(result["velocity_observable"], 1)
        self.assertEqual(result["relation_observable"], 1)
        self.assertEqual(
            result["observable_divergence_types"],
            "position|velocity|relation",
        )

    def test_roc_auc(self) -> None:
        self.assertEqual(_roc_auc([0, 1], [0.0, 1.0]), 1.0)
        self.assertEqual(_roc_auc([0, 1], [1.0, 0.0]), 0.0)
        self.assertEqual(_roc_auc([0, 1], [1.0, 1.0]), 0.5)

    def test_auc_views_do_not_mix_positive_background_into_primary(self) -> None:
        views = _frame_auc_views(
            normal_scores=[0.9, 1.0],
            positive_scores=[0.0, 0.1, 0.8],
            positive_labels=[0, 0, 1],
        )

        self.assertEqual(views["independent_normal"], 0.0)
        self.assertEqual(views["within_event"], 1.0)
        self.assertEqual(views["combined_legacy"], 0.5)

    def test_ewma_preserves_sustained_causal_surprise(self) -> None:
        first = _ewma_update(0.0, 10.0, alpha=0.5)
        second = _ewma_update(first, 10.0, alpha=0.5)

        self.assertEqual(first, 5.0)
        self.assertEqual(second, 7.5)

    def test_paired_counterfactual_diagnostics_align_event_and_frame(self) -> None:
        normal_frames = [
            self._paired_frame(position, 1.0, split="validation_normal")
            for position in reversed(range(4))
        ]
        counterfactual_scores = [1.0, 2.0, 3.0, 5.0]
        frames_to_anchor = [12, 11, 10, 0]
        positive_frames = [
            self._paired_frame(
                position,
                score,
                split="validation_positive",
                original_alert=int(position == 2),
                frames_to_anchor=frames_to_anchor[position],
            )
            for position, score in enumerate(counterfactual_scores)
        ]

        result = _paired_counterfactual_diagnostics(
            normal_frames=normal_frames,
            positive_frames=positive_frames,
            split="validation_positive",
        )

        self.assertEqual(len(result["frames"]), 4)
        self.assertEqual(result["frames"][2]["paired_causal_ewma_delta"], 2.0)
        pre_anchor = result["summaries"]["early_immune_pre_anchor_10"]
        self.assertEqual(pre_anchor["matched_frame_count"], 4)
        self.assertEqual(pre_anchor["missing_normal_frame_count"], 0)
        self.assertEqual(pre_anchor["pre_window_frame_count"], 2)
        self.assertEqual(pre_anchor["post_window_frame_count"], 1)
        self.assertEqual(pre_anchor["paired_delta_auroc"], 1.0)
        self.assertLess(
            pre_anchor["paired_delta_auroc_all_background"],
            pre_anchor["paired_delta_auroc"],
        )
        self.assertEqual(pre_anchor["target_scored_frame_count"], 4)

    def test_dual_track_label_windows(self) -> None:
        frames = [
            {"original_alert": 0, "frames_to_anchor": 4},
            {"original_alert": 0, "frames_to_anchor": 3},
            {"original_alert": 1, "frames_to_anchor": 2},
            {"original_alert": 0, "frames_to_anchor": 1},
            {"original_alert": 0, "frames_to_anchor": 0},
        ]

        self.assertEqual(
            _label_window_mask_from_frames(frames, "original"),
            [False, False, True, False, False],
        )
        self.assertEqual(
            _label_window_mask_from_frames(frames, "original_dilate_1"),
            [False, True, True, True, False],
        )
        self.assertEqual(
            _label_window_mask_from_frames(frames, "pre_anchor_3"),
            [False, True, True, True, False],
        )

    def test_top_edge_hit_type(self) -> None:
        self.assertEqual(
            _top_edge_hit_type(
                top_edge="1->2",
                target_source="1",
                target_target="2",
            ),
            "exact_directed",
        )
        self.assertEqual(
            _top_edge_hit_type(
                top_edge="2->1",
                target_source="1",
                target_target="2",
            ),
            "exact_reverse",
        )
        self.assertEqual(
            _top_edge_hit_type(
                top_edge="1->3",
                target_source="1",
                target_target="2",
            ),
            "one_target_node",
        )

    @staticmethod
    def _paired_frame(
        position: int,
        score: float,
        *,
        split: str,
        original_alert: int = 0,
        frames_to_anchor: int = -1,
    ) -> dict:
        return {
            "split": split,
            "event_index": 0,
            "event_uid": "event-a",
            "scenario_id": f"{split}-event-a",
            "frame_position": position,
            "frames_to_anchor": frames_to_anchor,
            "original_alert": original_alert,
            "physical": int(frames_to_anchor == 0),
            "aggregations": {
                "ego_incident_causal_raw": score,
                "ego_incident_causal_ewma": score,
            },
            "target_pair_best_score": score / 2.0,
        }


if __name__ == "__main__":
    unittest.main()
