# Relational Immune Recognition (RIR)

Code accompanying **Safe-Only Learning of Road-User Relations for Hazardous Interaction Discovery**.

This source release contains safe-only relation learning, family-disjoint calibration and evaluation, physical and learned comparators, DeepAccident adaptation, and existing unit tests. It is a source release: raw datasets, graph caches, trained weights, and the historical frozen model bundle are not included. Do not interpret a successful test run as reproduction of all paper experiments.

## Installation and tests

Use Python 3.10 in a fresh environment:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The versions are the observed local experiment environment. Installation on another machine has not been verified; install the appropriate PyTorch build for your platform if needed. The algorithm tests use synthetic fixtures and do not require downloading driving data.

## Data and reproduction

See [REPRODUCING.md](REPRODUCING.md) for the actual pipeline and artifact prerequisites, [DATA.md](DATA.md) for dataset placement and split identities, and [CODE_MAP.md](CODE_MAP.md) for entry points. Run all commands from this repository root. Some historical analysis scripts are intended as executable modules and have top-level execution; do not import every module indiscriminately.

`configs/riskbench_family_splits.json` records the existing paper split identities, without machine-specific paths. `reference_results/` contains small original result tables to compare with reruns; these tables are not raw training data. No test labels may be used for model, threshold, or parameter selection.

## Interpretation

RIR is most clearly supported as complementary relational evidence beyond the fixed TTC reference on RiskBench. Standalone evaluation retains inverse-TTC features and CPA geometry. Both evaluation datasets are simulated. Frozen DeepAccident detection does not establish additional recall beyond its physical reference, real-world transfer, reviewer time savings, or downstream safety gains.

## Licensing and attribution

Vendored Social-GAN files retain their upstream MIT license and attribution. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No license has been selected for the original project code; the authors should choose one before inviting third-party reuse. Dataset licenses remain with their respective providers.

Local compatibility links for data/results and legacy source modules may exist in the author's working folder. They are ignored by Git and are not required by a fresh checkout. Push only the contents tracked from this directory, not its parent research project.

## Submission preparation update 2026-11-25

The source now includes `02_alt_temporal_train.py` and `relation_schema.py`, which the training wrapper loads dynamically by file path. These were absent from the earlier public export even though its unit tests passed. The original algorithms and weights are unchanged.

```bash
python tools/verify_source.py
python -m pytest -q
```

A separate asset, `rir_frozen_models_20260908_v2.zip`, contains the original frozen package with 15 checkpoints, calibration distributions, the source snapshot and its integrity manifest. Extract its top-level directory under `results/`, then run:

```bash
python tools/check_frozen_models.py
```

The asset is prepared for a GitHub Release and is not automatically published by this source archive. Some historical records retain workstation paths for integrity; full dataset replay still needs separately acquired RiskBench/DeepAccident data and path-aware setup. Fresh-training commands in REPRODUCING.md rebuild portable run artifacts. No claim of a new full experiment rerun is made by packaging.
