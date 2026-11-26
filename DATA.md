# Data and split identities

Acquire RiskBench and DeepAccident from their official providers and follow their terms. Raw data is not redistributed here. Project links:

- RiskBench: https://github.com/HCIS-Lab/RiskBench
- DeepAccident: https://deepaccident.github.io/

RiskBench metadata is expected under `data/riskbench/raw/metadata/`, including `state/` and the annotations consumed by the adapter. The family-index builder accepts `--metadata-root` for another location. Inspect `experiments/ego_immune_recognition/riskbench_metadata/adapter.py` and the metadata fixtures in the tests for the exact schema.

The paper uses 666 scenario families: 246 safe-core, 273 hard-safe, 147 collision. Whole families are kept together. The existing representative IDs and fold roles are recorded in `configs/riskbench_family_splits.json`; check rebuilt indexes against these identities rather than assuming that a different dataset version gives the same split.

RiskBench provides collision-anchored positive labels. Recall does not cover every resolved non-collision hazardous interaction. Candidate budgets are calibrated on safe validation data, not selected from test labels. An additional-candidate rate uses the reference-negative safe subset; standalone FAR uses all hard-safe events.

DeepAccident requires plain-text labels/meta for accident and matched normal runs, and a paired focal-view manifest. The paper's 39-family test list is in `configs/deepaccident_external_families.txt`. That historical batch is no longer prospectively untouched. `prepare_second_batch` is a historical selection utility, not a command to relabel an already inspected batch as new. Supply `--root` to the DeepAccident audit for your dataset location.

Historical local indexes/receipts contain absolute paths. Do not edit sealed freeze manifests or rewrite their hashes. For fresh training, rebuild indexes under your current repository root. Historical replay additionally requires the original caches, trained checkpoints, and complete frozen bundle; these are not in this source release.
