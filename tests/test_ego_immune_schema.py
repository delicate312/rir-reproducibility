from __future__ import annotations

import unittest

import numpy as np

from experiments.ego_immune_recognition import (
    AgentType,
    EgoFrame,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    RoadUserState,
    Scenario,
    build_road_user_graph,
)


class EgoImmuneSchemaTest(unittest.TestCase):
    def make_frame(self) -> EgoFrame:
        return EgoFrame(
            scenario_id="toy_scenario",
            frame_index=0,
            timestamp_s=0.0,
            ego_agent_id="ego",
            road_users=[
                RoadUserState(
                    agent_id="ego",
                    agent_type=AgentType.VEHICLE,
                    position_xy_m=(0.0, 0.0),
                    velocity_xy_mps=(8.0, 0.0),
                    heading_rad=0.0,
                    length_m=4.5,
                    width_m=1.8,
                    height_m=1.6,
                ),
                RoadUserState(
                    agent_id="ped_1",
                    agent_type=AgentType.PEDESTRIAN,
                    position_xy_m=(12.0, 2.0),
                    velocity_xy_mps=(-1.0, 0.0),
                    heading_rad=3.14,
                    length_m=0.5,
                    width_m=0.5,
                    height_m=1.7,
                    observation_confidence=0.7,
                ),
                RoadUserState(
                    agent_id="far_cyclist",
                    agent_type=AgentType.CYCLIST,
                    position_xy_m=(120.0, 0.0),
                    velocity_xy_mps=(3.0, 0.0),
                    observation_confidence=0.3,
                ),
                RoadUserState(
                    agent_id="occluded_vehicle",
                    agent_type=AgentType.VEHICLE,
                    position_xy_m=(20.0, 0.0),
                    velocity_xy_mps=(0.0, 0.0),
                    observed=False,
                    observation_confidence=0.0,
                ),
            ],
        )

    def test_scenario_accepts_mixed_road_users(self) -> None:
        scenario = Scenario(
            scenario_id="toy_scenario",
            dataset_name="unit_test",
            frames=[self.make_frame()],
        )
        self.assertEqual(scenario.frame_count, 1)
        self.assertIn(AgentType.PEDESTRIAN, scenario.road_user_types)
        self.assertIn(AgentType.CYCLIST, scenario.road_user_types)

    def test_graph_builder_is_ego_centric_and_partial(self) -> None:
        graph = build_road_user_graph(
            self.make_frame(),
            max_observation_range_m=50.0,
            edge_radius_m=30.0,
            include_unobserved=False,
        )

        self.assertEqual(graph.ego_agent_id, "ego")
        self.assertEqual(graph.node_ids, ("ego", "ped_1"))
        self.assertEqual(graph.node_features.shape[1], len(NODE_FEATURE_NAMES))
        self.assertEqual(graph.edge_features.shape[1], len(EDGE_FEATURE_NAMES))
        self.assertTrue(np.all(graph.observed_mask))
        self.assertGreater(graph.edge_count, 0)

        pedestrian_index = graph.node_ids.index("ped_1")
        is_pedestrian_feature = NODE_FEATURE_NAMES.index("is_pedestrian")
        self.assertEqual(
            graph.node_features[pedestrian_index, is_pedestrian_feature],
            1.0,
        )

    def test_unobserved_agents_can_be_kept_for_diagnostic_labels(self) -> None:
        graph = build_road_user_graph(
            self.make_frame(),
            max_observation_range_m=50.0,
            edge_radius_m=30.0,
            include_unobserved=True,
        )

        self.assertIn("occluded_vehicle", graph.node_ids)
        occluded_index = graph.node_ids.index("occluded_vehicle")
        self.assertFalse(bool(graph.observed_mask[occluded_index]))


if __name__ == "__main__":
    unittest.main()
