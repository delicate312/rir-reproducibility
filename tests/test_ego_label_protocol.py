from __future__ import annotations

import unittest

import numpy as np

from experiments.ego_immune_recognition import (
    AgentType,
    EarlyDangerLabelConfig,
    EgoFrame,
    RoadUserState,
    build_road_user_graph,
    label_early_danger_sequence,
)


class EgoLabelProtocolTest(unittest.TestCase):
    def make_graph_sequence(self):
        graphs = []
        for index, other_x in enumerate([20.0, 12.0, 4.0]):
            frame = EgoFrame(
                scenario_id="label_toy",
                frame_index=index,
                timestamp_s=index * 0.1,
                ego_agent_id="ego",
                road_users=[
                    RoadUserState(
                        agent_id="ego",
                        agent_type=AgentType.VEHICLE,
                        position_xy_m=(0.0, 0.0),
                        velocity_xy_mps=(10.0, 0.0),
                        heading_rad=0.0,
                    ),
                    RoadUserState(
                        agent_id="lead",
                        agent_type=AgentType.VEHICLE,
                        position_xy_m=(other_x, 0.0),
                        velocity_xy_mps=(0.0, 0.0),
                        heading_rad=0.0,
                    ),
                ],
            )
            graphs.append(
                build_road_user_graph(
                    frame,
                    max_observation_range_m=50.0,
                    edge_radius_m=50.0,
                )
            )
        return graphs

    def test_labels_alert_before_physical_anchor(self) -> None:
        labels = label_early_danger_sequence(
            self.make_graph_sequence(),
            EarlyDangerLabelConfig(
                physical_inverse_ttc_threshold_per_s=2.0,
                critical_distance_m=1.0,
                alert_lead_frames=2,
                alert_horizon_frames=1,
            ),
        )

        self.assertTrue(labels.has_physical_danger)
        self.assertEqual(labels.anchor_frame_position, 2)
        self.assertTrue(labels.frame_labels[0].is_immune_alert)
        self.assertTrue(labels.frame_labels[1].is_immune_alert)
        self.assertFalse(labels.frame_labels[2].is_immune_alert)
        self.assertTrue(labels.frame_labels[2].is_physical_danger)

    def test_normal_sequence_has_no_alert(self) -> None:
        graphs = self.make_graph_sequence()[:1]
        labels = label_early_danger_sequence(
            graphs,
            EarlyDangerLabelConfig(
                physical_inverse_ttc_threshold_per_s=10.0,
                critical_distance_m=1.0,
            ),
        )

        self.assertFalse(labels.has_physical_danger)
        self.assertEqual(labels.immune_alert_count, 0)
        self.assertEqual(labels.physical_danger_count, 0)

    def test_presented_edges_are_ego_related_by_default(self) -> None:
        labels = label_early_danger_sequence(self.make_graph_sequence())
        presented = labels.frame_labels[-1].top_presented_edges

        self.assertGreater(len(presented), 0)
        self.assertTrue(
            all("ego" in edge[:2] for edge in presented)
        )
        self.assertTrue(
            all(np.isfinite(edge[2]) for edge in presented)
        )


if __name__ == "__main__":
    unittest.main()
