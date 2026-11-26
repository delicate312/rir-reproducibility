from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from experiments.ego_immune_recognition.rule_baseline_dual_track_eval import (
    rule_scores_for_graph,
)


class RuleBaselineDualTrackEvalTest(unittest.TestCase):
    def test_rule_scores_use_all_and_ego_incident_edges(self) -> None:
        graph = SimpleNamespace(
            ego_agent_id="2",
            node_ids=["1", "2", "3"],
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            edge_attr=torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 10.0, 2.0, 0.2, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 5.0, 4.0, 0.8, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 7.0, 3.0, 1.0, 1.0],
                ]
            ),
        )

        scores = rule_scores_for_graph(graph)

        self.assertEqual(scores["inverse_ttc_all_max"], 3.0)
        self.assertAlmostEqual(scores["inverse_ttc_ego_max"], 0.8)
        self.assertAlmostEqual(scores["inverse_distance_all_max"], 1.0)
        self.assertAlmostEqual(scores["inverse_distance_ego_max"], 0.2)
        self.assertEqual(scores["closing_speed_all_max"], 7.0)
        self.assertEqual(scores["closing_speed_ego_max"], 4.0)


if __name__ == "__main__":
    unittest.main()
