# Code map

All module paths below start with `experiments.ego_immune_recognition`.

| Role | Module |
| --- | --- |
| Observation schema, features | `schema`, `observation_model`, `pyg_packager` |
| Temporal relation forecaster | `temporal_relation_train_eval` |
| RiskBench adapter and family splits | `riskbench_metadata.adapter`, `family_grouped_nested_cv` |
| Safe-only three-seed training | `riskbench_metadata.pair_damp_normal_only_multiseed` |
| Hard-safe nested parameter selection | `riskbench_metadata.damp_hard_safe_grid_selection` |
| Complementary detection and mechanism controls | `riskbench_metadata.damp_decision_necessity_ablation` |
| Standalone all-safe selection | `riskbench_metadata.standalone_rir_audit` |
| Physical/history controls | `riskbench_metadata.history_geometry_validation`, `probabilistic_rollout_evaluation` |
| Selected RIR localization | `riskbench_metadata.selected_rir_localization` |
| Timing and review ranking | `riskbench_metadata.fine_early_recognition`, `review_burden_analysis` |
| Safe Social-GAN retraining | `riskbench_metadata.social_gan_verified_runner`, `social_gan_convergence_extension` |
| Social-GAN evaluation/statistics | `riskbench_metadata.social_gan_verified_evaluation_runner`, `verified_strong_baseline_statistics`, `verified_standalone_statistics` |
| Frozen artifact checks and scoring | `riskbench_metadata.prospective_validation` |
| DeepAccident audit/adapter | `deepaccident.mini_audit`, `deepaccident.adapter` |
| DeepAccident frozen evaluation | `deepaccident.frozen_pilot`, `physical_union_validation` |

Some dependency modules retain historical names (ablation, exploratory, v1/v2). They are included because the final scripts import their helper functions, not because all of their historical experimental claims are part of the final paper. Generic project README files, obsolete AV2/US101 branches, audit servers, videos, manuscript files, and large cached results are excluded.
