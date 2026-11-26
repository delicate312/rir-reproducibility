from __future__ import annotations

import unittest

from experiments.ego_immune_recognition import (
    AgentType,
    EarlyDangerLabelConfig,
    EgoFrame,
    RoadUserState,
    build_road_user_graph,
    label_early_danger_sequence,
)
from experiments.ego_immune_recognition.pyg_packager import package_pyg_sequence


class EgoPyGPackagerTest(unittest.TestCase):
    def test_package_sequence_carries_labels(self) -> None:
        frame = EgoFrame(
            scenario_id="pyg_toy",
            frame_index=0,
            timestamp_s=0.0,
            ego_agent_id="ego",
            road_users=[
                RoadUserState(
                    agent_id="ego",
                    agent_type=AgentType.VEHICLE,
                    position_xy_m=(0.0, 0.0),
                    velocity_xy_mps=(10.0, 0.0),
                ),
                RoadUserState(
                    agent_id="lead",
                    agent_type=AgentType.VEHICLE,
                    position_xy_m=(2.0, 0.0),
                    velocity_xy_mps=(0.0, 0.0),
                ),
            ],
        )
        graph = build_road_user_graph(frame, edge_radius_m=10.0)
        labels = label_early_danger_sequence(
            [graph],
            EarlyDangerLabelConfig(
                physical_inverse_ttc_threshold_per_s=2.0,
                critical_distance_m=1.0,
            ),
        )
        package = package_pyg_sequence([graph], labels)

        self.assertEqual(package.scenario_id, "pyg_toy")
        self.assertEqual(len(package.graphs), 1)
        data = package.graphs[0]
        self.assertEqual(data.x.shape[0], graph.node_count)
        self.assertEqual(data.edge_attr.shape[0], graph.edge_count)
        self.assertEqual(int(data.y_physical_danger.item()), 1)
        self.assertTrue(hasattr(data, "edge_presentation_mask"))


if __name__ == "__main__":
    unittest.main()
