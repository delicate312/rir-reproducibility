"""Load eager PyG packages or indexed lazy graph datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def load_graph_dataset(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.suffix == ".pt":
        return torch.load(resolved, weights_only=False)
    if resolved.suffix != ".json":
        raise ValueError(f"Unsupported graph dataset format: {resolved}")

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("format") != "riskbench-metadata-lazy-v1":
        raise ValueError(f"Unsupported lazy dataset protocol: {resolved}")
    from .riskbench_metadata.lazy_dataset import load_lazy_package

    return load_lazy_package(resolved, payload=payload)
