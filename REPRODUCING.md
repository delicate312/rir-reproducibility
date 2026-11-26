# Reproduction stages

## 1. Build the family-disjoint RiskBench indexes

Acquire the metadata first, then run:

```bash
python -m experiments.ego_immune_recognition.riskbench_metadata.family_grouped_nested_cv --build-only --metadata-root data/riskbench/raw/metadata
```

Compare the created fold identities with `configs/riskbench_family_splits.json`. The five outer folds and split seed 20260728 are part of the existing protocol. The source builder will create lazy graph caches as they are consumed.

## 2. Train and freeze checkpoint selection

```bash
python -m experiments.ego_immune_recognition.riskbench_metadata.pair_damp_normal_only_multiseed --seeds 20260812,20260813,20260814 --epochs 3 --eval-epochs 1,3
```

Checkpoint selection uses hard-safe objective loss. Do not use collision labels for training or selection. The outputs include per-seed checkpoint receipts and causal frame caches needed by the next stages. Existing historical receipts with another machine's absolute paths are not automatically portable; fresh outputs use the current root.

## 3. Complementary parameter selection and evaluation

```bash
python -m experiments.ego_immune_recognition.riskbench_metadata.damp_hard_safe_grid_selection --fpr-budget 0.05
python -m experiments.ego_immune_recognition.riskbench_metadata.damp_decision_necessity_ablation --fpr-budget 0.05
```

These defaults reconstruct the conservative 5% validation setting. To run 10%, provide separate `--output-root` directories to the selector/evaluator and pass the corresponding selector output through `--selection-root`; never reuse a 5% selection folder as a 10% result. Use module `--help` for its supported arguments. The observed test candidate rate need not equal the validation budget.

## 4. Standalone, timing, ranking and baselines

```bash
python -m experiments.ego_immune_recognition.riskbench_metadata.standalone_rir_audit
python -m experiments.ego_immune_recognition.riskbench_metadata.standalone_rir_report
```

These use the regenerated indexes, model/frame caches and all-safe validation selection. Social-GAN's final safe-training configuration is in `social_gan_verified_runner`; run the runner followed by `social_gan_convergence_extension` and `social_gan_verified_evaluation_runner`, then the verified statistics modules. The original source configuration has three training seeds (20260913–20260915) and five folds. Timing/localization/ranking modules require their upstream result files; consult their source constants and CODE_MAP rather than treating them as independent training entry points.

## 5. Frozen DeepAccident transfer

This stage requires the complete historical bundle `results/rir_prospective_freeze_20260908_v2` (15 checkpoints, calibration arrays, protocol and hash manifest), the DeepAccident paired-view manifest, and the corresponding raw labels/meta. They are not redistributed here. `deepaccident.frozen_pilot --help` documents the manifest/protocol/bundle arguments. Never reconstruct a historical freeze and call it the original preregistered artifact, or refit thresholds on external test outcomes. The exact 39-family identity list is provided separately.

## Verification limits

The release is checked with the existing unit tests in an isolated source-only export. Full model training, raw-data reconstruction, platform installation, and exact end-to-end numerical reproduction have not been rerun as part of packaging. Small reference CSVs are supplied for comparison; importing executable runner modules can launch work or change run files, so use explicit commands only.
