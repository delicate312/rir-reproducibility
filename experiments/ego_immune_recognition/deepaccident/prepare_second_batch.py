"""Lock a never-scored DeepAccident validation batch before model inference."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from ..riskbench_metadata.prospective_validation import digest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FULL_MANIFEST = (
    PROJECT_ROOT
    / "data/processed/ego_immune_deepaccident_val_complete_v1"
    / "deepaccident_val_complete_focal_view_manifest.csv"
)
DEFAULT_PRIOR_MANIFEST = (
    PROJECT_ROOT
    / "data/processed/ego_immune_deepaccident_val01_v1"
    / "deepaccident_val01_focal_view_manifest.csv"
)
DEFAULT_BUNDLE = PROJECT_ROOT / "results/rir_prospective_freeze_20260908_v2"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data/processed/ego_immune_deepaccident_val02_new_v1"
)


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot lock an empty validation batch")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _family_digest(families: list[str]) -> str:
    payload = "".join(f"{family}\n" for family in families).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", type=Path, default=DEFAULT_FULL_MANIFEST)
    parser.add_argument("--prior-manifest", type=Path, default=DEFAULT_PRIOR_MANIFEST)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    full_manifest = args.full_manifest.expanduser().resolve()
    prior_manifest = args.prior_manifest.expanduser().resolve()
    bundle = args.bundle.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise ValueError(f"Refuse to overwrite a locked batch: {output_root}")

    full_rows = [row for row in _read(full_manifest) if _truthy(row["usable"])]
    prior_rows = [row for row in _read(prior_manifest) if _truthy(row["usable"])]
    prior_families = {row["pair_key"] for row in prior_rows}
    selected = [row for row in full_rows if row["pair_key"] not in prior_families]
    families = sorted({row["pair_key"] for row in selected})
    if {row["pair_key"] for row in selected} & prior_families:
        raise AssertionError("Previously scored family leaked into the new batch")

    output_root.mkdir(parents=True)
    manifest = output_root / "deepaccident_val02_new_focal_view_manifest.csv"
    family_list = output_root / "eligible_families.txt"
    protocol_path = output_root / "external_validation_protocol_v1.json"
    _write_csv(manifest, selected)
    family_list.write_text(
        "".join(f"{family}\n" for family in families), encoding="utf-8"
    )

    protocol = {
        "protocol": "deepaccident-val02-new-families-frozen-rir-external-validation-v1",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "status_at_lock": (
            "No frozen RIR or physical-reference score had been computed for any "
            "family in this batch."
        ),
        "purpose": (
            "Second-batch independent evaluation on every eligible DeepAccident "
            "validation family not present in the previously scored val01 manifest."
        ),
        "full_audit_manifest_path": str(full_manifest),
        "full_audit_manifest_sha256": digest(full_manifest),
        "prior_scored_manifest_path": str(prior_manifest),
        "prior_scored_manifest_sha256": digest(prior_manifest),
        "input_manifest_sha256": digest(manifest),
        "eligible_family_list_sha256": _family_digest(families),
        "frozen_bundle_manifest_sha256": (
            bundle / "MANIFEST.sha256"
        ).read_text(encoding="utf-8").strip(),
        "eligible_view_count": len(selected),
        "eligible_family_count": len(families),
        "eligible_views": (
            "All collision-involved focal views from valid paired families in the "
            "complete DeepAccident validation split, excluding every family already "
            "scored in the val01 first batch."
        ),
        "exclusion_rule": (
            "Exclude invalid collision metadata and families already present in the "
            "locked val01 evaluation; do not exclude a family or view using model scores."
        ),
        "family_unit": (
            "Scenario pair_key; focal views and three frozen model seeds do not count "
            "as additional independent families."
        ),
        "fold_routing": (
            "Frozen prospective-validation-v1 SHA-256 family routing, identical across "
            "all focal views and all three seeds."
        ),
        "model": (
            "The 15 checkpoints and hard-safe-only calibration artifacts sealed in "
            "rir_prospective_freeze_20260908_v2."
        ),
        "primary_operating_point": "Frozen standalone 10% all-hard-safe review budget.",
        "secondary_operating_point": "Frozen standalone 5% all-hard-safe review budget.",
        "physical_reference": (
            "The fixed RiskBench validation-normal empirical inverse-TTC percentile, "
            "with the previously specified 0.95 activation threshold."
        ),
        "guarded_decision": "Logical union of the physical reference and RIR relation alert.",
        "adaptation": {
            "native_rate": "10 Hz",
            "model_rate": "20 Hz",
            "rule": (
                "Keep each observed state and insert one causal half-step extrapolated "
                "from current position and current/past-derived velocity."
            ),
            "future_use": (
                "No next observation, future trajectory, collision identity, collision "
                "outcome, or matched-normal outcome enters model features."
            ),
            "end_rule": "No state is appended after the final recorded observation.",
            "input_window": "Final 40 model-clock frames, matching the frozen contract.",
        },
        "evaluation_window": (
            "Observed frames 30 to 1 model frames before the independent collision anchor; "
            "because recordings end 1.0 s before collision, the available overlap is "
            "approximately 1.5 to 1.0 s before collision."
        ),
        "positive_label": "Accident member with valid independent collision metadata.",
        "negative_label": (
            "Matched normal realization of the same family, aligned to an equal observation window."
        ),
        "primary_methods": [
            "physical_reference",
            "rir_all_safe_selected",
            "physical_or_rir",
            "cpa_peak",
            "raw_residual_hybrid",
        ],
        "reported_metrics": [
            "overall accident recall for physical, relation, and union decisions",
            "RIR rescue among physical-reference-negative accidents",
            "physical and additional relation candidate rates on matched normals",
            "paired accident-normal outcomes and score margins",
            "lead time on the observed evaluation interval",
            "collision-partner top-1 localization where the target is observable",
        ],
        "statistics": (
            "Family-level descriptive estimates and family bootstrap intervals; the family "
            "is the independent unit and seeds quantify frozen-model variation only."
        ),
        "summary_title": "DeepAccident second-batch frozen external validation",
        "interpretation_boundary": (
            "This batch is independent of result inspection at family level. The complete-split "
            "aggregate that combines it with val01 is descriptive because val01 was previously viewed."
        ),
        "completion_status": "completed_external_validation_second_batch",
    }
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))
    print(f"manifest: {manifest}")
    print(f"family list: {family_list}")
    print(f"protocol: {protocol_path}")


if __name__ == "__main__":
    main()
