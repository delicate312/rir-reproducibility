import unittest
from types import SimpleNamespace

import torch

from experiments.ego_immune_recognition.danger_signal.diagnostics import (
    apply_self_context_inhibition,
    directional_damage,
    edge_danger_components,
    longest_true_run,
)
from experiments.ego_immune_recognition.schema import EDGE_FEATURE_NAMES


class DangerSignalDiagnosticsTest(unittest.TestCase):
    def graph(self):
        return SimpleNamespace(
            edge_attr=torch.zeros((2, len(EDGE_FEATURE_NAMES))),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            node_ids=["ego", "other"],
            ego_agent_id="ego",
        )

    def test_directional_damage_only_counts_dangerous_error_direction(self):
        graph = self.graph()
        expected = torch.zeros_like(graph.edge_attr)
        graph.edge_attr[0, EDGE_FEATURE_NAMES.index("distance_m")] = -3.0
        graph.edge_attr[0, EDGE_FEATURE_NAMES.index("closing_speed_mps")] = 1.5
        graph.edge_attr[0, EDGE_FEATURE_NAMES.index("inverse_ttc_per_s")] = 0.6
        graph.edge_attr[1, EDGE_FEATURE_NAMES.index("distance_m")] = 3.0
        graph.edge_attr[1, EDGE_FEATURE_NAMES.index("closing_speed_mps")] = -1.5
        graph.edge_attr[1, EDGE_FEATURE_NAMES.index("inverse_ttc_per_s")] = -0.6

        damage = directional_damage(
            graph, expected, torch.tensor([True, True])
        )

        self.assertAlmostEqual(
            float(damage[0]), (3.0 + 1.5 + 0.6) / 3.0, places=6
        )
        self.assertEqual(float(damage[1]), 0.0)

    def test_edge_state_accumulates_then_emits_recovery(self):
        graph = self.graph()
        previous = {}
        first = edge_danger_components(
            graph=graph,
            causal_scores=torch.tensor([1.0, 0.0]),
            damage_scores=torch.tensor([0.5, 0.0]),
            coverage=torch.tensor([True, True]),
            previous=previous,
            persistence_decay=0.8,
        )
        second = edge_danger_components(
            graph=graph,
            causal_scores=torch.tensor([2.0, 0.0]),
            damage_scores=torch.tensor([1.0, 0.0]),
            coverage=torch.tensor([True, True]),
            previous=previous,
            persistence_decay=0.8,
        )
        third = edge_danger_components(
            graph=graph,
            causal_scores=torch.tensor([0.5, 0.0]),
            damage_scores=torch.tensor([0.2, 0.0]),
            coverage=torch.tensor([True, True]),
            previous=previous,
            persistence_decay=0.8,
        )

        self.assertAlmostEqual(float(first["persistence"][0]), 0.1)
        self.assertGreater(float(second["stress"][0]), 0.0)
        self.assertGreater(float(second["persistence"][0]), 0.1)
        self.assertGreater(float(third["recovery"][0]), 0.0)
        self.assertGreater(float(third["persistence"][0]), 0.0)

    def test_recovery_waits_for_relation_initialization_predictions(self):
        graph = self.graph()
        previous = {}
        covered_counts = {}

        outputs = []
        for causal, damage in [
            (1.0, 0.5),
            (2.0, 1.0),
            (0.5, 0.2),
            (0.2, 0.1),
        ]:
            outputs.append(
                edge_danger_components(
                    graph=graph,
                    causal_scores=torch.tensor([causal, 0.0]),
                    damage_scores=torch.tensor([damage, 0.0]),
                    coverage=torch.tensor([True, True]),
                    previous=previous,
                    persistence_decay=0.8,
                    covered_observation_counts=covered_counts,
                    recovery_initialization_predictions=3,
                )
            )

        self.assertEqual(float(outputs[2]["recovery"][0]), 0.0)
        self.assertGreater(float(outputs[2]["onset_shock"][0]), 0.0)
        self.assertGreater(float(outputs[3]["recovery"][0]), 0.0)
        self.assertEqual(float(outputs[3]["onset_shock"][0]), 0.0)
        self.assertEqual(covered_counts["ego->other"], 4)

    def test_negative_recovery_initialization_predictions_is_rejected(self):
        with self.assertRaises(ValueError):
            edge_danger_components(
                graph=self.graph(),
                causal_scores=torch.tensor([1.0, 0.0]),
                damage_scores=torch.tensor([0.5, 0.0]),
                coverage=torch.tensor([True, True]),
                previous={},
                recovery_initialization_predictions=-1,
            )

    def test_longest_true_run(self):
        self.assertEqual(
            longest_true_run([False, True, True, False, True, True, True]),
            3,
        )

    def test_self_context_uses_prefix_only(self):
        frames = [
            {
                "event_index": 0,
                "frame_position": index,
                "aggregations": {
                    "danger_activation_recovery_only": value,
                },
            }
            for index, value in enumerate([1.0, 3.0, 8.0])
        ]

        apply_self_context_inhibition(frames, warmup_frames=2)

        self.assertEqual(frames[0]["aggregations"]["danger_self_context"], 2.0)
        self.assertEqual(
            frames[0]["aggregations"]["danger_activation_inhibited"], 0.0
        )
        self.assertEqual(
            frames[2]["aggregations"]["danger_activation_inhibited"], 6.0
        )


if __name__ == "__main__":
    unittest.main()
