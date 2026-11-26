from __future__ import annotations

import unittest

import numpy as np

from experiments.ego_immune_recognition import (
    AgentType,
    EgoFrame,
    ObservationModelConfig,
    RoadUserState,
    apply_ego_observation_model,
    build_road_user_graph,
)


class EgoObservationModelTest(unittest.TestCase):
    def make_frame(self) -> EgoFrame:
        return EgoFrame(
            scenario_id="visibility_toy",
            frame_index=0,
            timestamp_s=0.0,
            ego_agent_id="ego",
            road_users=[
                RoadUserState(
                    agent_id="ego",
                    agent_type=AgentType.VEHICLE,
                    position_xy_m=(0.0, 0.0),
                    velocity_xy_mps=(0.0, 0.0),
                    heading_rad=0.0,
                    length_m=4.5,
                    width_m=1.8,
                ),
                RoadUserState(
                    agent_id="near_vehicle",
                    agent_type=AgentType.VEHICLE,
                    position_xy_m=(10.0, 0.0),
                    velocity_xy_mps=(1.0, 0.0),
                    length_m=4.5,
                    width_m=2.0,
                ),
                RoadUserState(
                    agent_id="hidden_pedestrian",
                    agent_type=AgentType.PEDESTRIAN,
                    position_xy_m=(20.0, 0.2),
                    velocity_xy_mps=(0.0, 0.0),
                    length_m=0.5,
                    width_m=0.5,
                ),
                RoadUserState(
                    agent_id="behind_vehicle",
                    agent_type=AgentType.VEHICLE,
                    position_xy_m=(-10.0, 0.0),
                    velocity_xy_mps=(0.0, 0.0),
                ),
                RoadUserState(
                    agent_id="far_cyclist",
                    agent_type=AgentType.CYCLIST,
                    position_xy_m=(90.0, 0.0),
                    velocity_xy_mps=(2.0, 0.0),
                ),
            ],
        )

    def test_fov_range_and_occlusion_mark_unobserved_agents(self) -> None:
        degraded = apply_ego_observation_model(
            self.make_frame(),
            ObservationModelConfig(
                max_range_m=50.0,
                horizontal_fov_deg=120.0,
                occlusion_enabled=True,
                keep_unobserved=True,
            ),
        )

        by_id = {user.agent_id: user for user in degraded.road_users}
        self.assertTrue(by_id["ego"].observed)
        self.assertTrue(by_id["near_vehicle"].observed)
        self.assertFalse(by_id["hidden_pedestrian"].observed)
        self.assertFalse(by_id["behind_vehicle"].observed)
        self.assertFalse(by_id["far_cyclist"].observed)
        self.assertEqual(by_id["hidden_pedestrian"].observation_confidence, 0.0)

    def test_unobserved_agents_can_be_dropped_before_graph_building(self) -> None:
        degraded = apply_ego_observation_model(
            self.make_frame(),
            ObservationModelConfig(
                max_range_m=50.0,
                horizontal_fov_deg=120.0,
                occlusion_enabled=True,
                keep_unobserved=False,
            ),
        )
        graph = build_road_user_graph(degraded)

        self.assertEqual(graph.node_ids, ("ego", "near_vehicle"))
        self.assertTrue(np.all(graph.observed_mask))

    def test_distance_confidence_decreases_before_range_cutoff(self) -> None:
        degraded = apply_ego_observation_model(
            self.make_frame(),
            ObservationModelConfig(
                max_range_m=100.0,
                full_confidence_range_m=20.0,
                min_visible_confidence=0.2,
                horizontal_fov_deg=360.0,
                occlusion_enabled=False,
            ),
        )

        by_id = {user.agent_id: user for user in degraded.road_users}
        self.assertEqual(
            by_id["near_vehicle"].observation_confidence,
            1.0,
        )
        self.assertGreater(
            by_id["far_cyclist"].observation_confidence,
            0.2,
        )
        self.assertLess(
            by_id["far_cyclist"].observation_confidence,
            1.0,
        )

    def test_distance_noise_is_reproducible_with_seeded_rng(self) -> None:
        config = ObservationModelConfig(
            max_range_m=100.0,
            position_noise_std_at_max_range_m=1.0,
            velocity_noise_std_at_max_range_mps=0.5,
            occlusion_enabled=False,
        )
        first = apply_ego_observation_model(
            self.make_frame(),
            config,
            rng=np.random.default_rng(7),
        )
        second = apply_ego_observation_model(
            self.make_frame(),
            config,
            rng=np.random.default_rng(7),
        )

        first_by_id = {user.agent_id: user for user in first.road_users}
        second_by_id = {user.agent_id: user for user in second.road_users}
        np.testing.assert_allclose(
            first_by_id["far_cyclist"].position_xy_m,
            second_by_id["far_cyclist"].position_xy_m,
        )


if __name__ == "__main__":
    unittest.main()
