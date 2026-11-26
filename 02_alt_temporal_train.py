"""
02_train_all_models.py

Train three unsupervised graph autoencoders on the final US-101 train-normal
snapshot graphs produced by 01e_package_graph_datasets.py.

Models
------
1. Vanilla_MLP_AE
   Node-wise MLP encoder. It does not use graph message passing.

2. GCN_AE
   Standard GCN encoder. It uses graph topology but ignores edge attributes in
   message passing.

3. Immune_GAE
   Edge-aware GATv2 encoder. It uses distance risk, TTC risk, and interaction
   strength during message passing, then reconstructs topology, edge attributes,
   and node state.

Training principle
------------------
Only train_normal graphs are used for optimization.
val_normal graphs are used for early stopping.
No OOD graph is used during training or model selection.

Outputs
-------
checkpoints/
├── Vanilla_MLP-AE_us101_temporal.pth
├── GCN-AE_us101_temporal.pth
└── Immune_GAE_us101_temporal.pth

results/training/
├── training_history.csv
├── training_summary.json
└── *_history.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from relation_schema import RELATION_SCHEMA_VERSION

try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATv2Conv, GCNConv
except ImportError as error:
    raise ImportError(
        "未安装 torch_geometric。请先在 immune_driving 环境中安装 PyTorch Geometric。"
    ) from error


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path("/Users/finn/ImmuneRecognition")

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "us101_final_graph_dataset_v2"
    / "us101_final_graph_dataset.pt"
)

DATASET_TAG = "us101"
DATASET_LABEL = "US-101"
EXPECTED_RELATION_SCHEMA_VERSION = RELATION_SCHEMA_VERSION

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "trail_v3_causal"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "training_trail_v3_causal"
)

SUMMARY_PATH = RESULT_DIR / "training_summary.json"
COMBINED_HISTORY_PATH = RESULT_DIR / "training_history.csv"

INTERHUB_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "interhub_trail_graph_dataset_v1"
    / "interhub_trail_graph_dataset.pt"
)
INTERHUB_RELATION_SCHEMA_VERSION = (
    "trail-interhub-source-centric-response-v1"
)
US101_EARLY_RESPONSE_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "us101_early_response_graph_dataset_v1"
    / "us101_early_response_graph_dataset.pt"
)
US101_EARLY_RESPONSE_RELATION_SCHEMA_VERSION = (
    "trail-us101-early-response-v1"
)
STRONG_INTERACTION_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "strong_interaction_trail_graph_dataset_v1"
    / "strong_interaction_trail_graph_dataset.pt"
)
STRONG_INTERACTION_RELATION_SCHEMA_VERSION = (
    "trail-strong-interaction-source-centric-response-v1"
)

DATASET_PROFILES = {
    "us101": {
        "dataset_path": DATASET_PATH,
        "dataset_tag": "us101",
        "dataset_label": "US-101",
        "expected_schema": RELATION_SCHEMA_VERSION,
        "checkpoint_dir": CHECKPOINT_DIR,
        "result_dir": RESULT_DIR,
    },
    "interhub": {
        "dataset_path": INTERHUB_DATASET_PATH,
        "dataset_tag": "interhub",
        "dataset_label": "InterHub counterfactual-response prototype",
        "expected_schema": INTERHUB_RELATION_SCHEMA_VERSION,
        "checkpoint_dir": (
            PROJECT_ROOT
            / "checkpoints"
            / "trail_interhub_v1"
        ),
        "result_dir": (
            PROJECT_ROOT
            / "results"
            / "training_trail_interhub_v1"
        ),
    },
    "us101-early-response": {
        "dataset_path": US101_EARLY_RESPONSE_DATASET_PATH,
        "dataset_tag": "us101_early_response",
        "dataset_label": (
            "US-101 natural braking/early delayed-response feasibility set"
        ),
        "expected_schema": (
            US101_EARLY_RESPONSE_RELATION_SCHEMA_VERSION
        ),
        "checkpoint_dir": (
            PROJECT_ROOT
            / "checkpoints"
            / "trail_us101_early_response_v1"
        ),
        "result_dir": (
            PROJECT_ROOT
            / "results"
            / "training_trail_us101_early_response_v1"
        ),
    },
    "strong-interaction": {
        "dataset_path": STRONG_INTERACTION_DATASET_PATH,
        "dataset_tag": "strong_interaction",
        "dataset_label": (
            "Strong-interaction counterfactual-response prototype"
        ),
        "expected_schema": (
            STRONG_INTERACTION_RELATION_SCHEMA_VERSION
        ),
        "checkpoint_dir": (
            PROJECT_ROOT
            / "checkpoints"
            / "trail_strong_interaction_v1"
        ),
        "result_dir": (
            PROJECT_ROOT
            / "results"
            / "training_trail_strong_interaction_v1"
        ),
    },
}


# ============================================================
# Reproducibility and training configuration
# ============================================================

RANDOM_SEED = 20260611

# Tiny local graphs usually train faster on CPU than on Apple MPS because MPS
# kernel-launch overhead dominates. Change to "auto" only if benchmarking proves
# MPS is faster on your machine.
DEVICE_MODE = "cpu"  # "cpu", "mps", "cuda", or "auto"

EPOCHS = 120
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

TEMPORAL_EPOCHS = 120
TEMPORAL_SEQUENCE_LENGTH = 21
TEMPORAL_SEQUENCE_STRIDE = 5
TEMPORAL_MAX_TIME_GAP_MS = 250
TEMPORAL_TRAIN_ON_FULL_CLIPS = True

RELATION_LOCALIZER_EPOCHS = 120
RELATION_MASK_START_FRACTION = 0.35
RELATION_MASK_END_FRACTION = 1.00
RELATION_MASK_CURRICULUM_EPOCHS = 80
RELATION_FULL_MASK_PROBABILITY = 0.15
RELATION_INFERENCE_MASK_GROUPS = 4
RELATION_MASKED_EDGE_WEIGHT = 3.0
RELATION_NEXT_EDGE_WEIGHT = 2.0
RELATION_MEMORYLESS_AUX_WEIGHT = 0.50
RELATION_COMPACTNESS_WEIGHT = 0.10
RELATION_PROTOTYPE_COMPACTNESS_WEIGHT = 0.25
RELATION_PROTOTYPE_SCORE_WEIGHT = 1.00
RELATION_PROTOTYPE_MOMENTUM = 0.95
RELATION_PROTOTYPE_MIN_SCALE = 0.25
RELATION_TOPOLOGY_WEIGHT = 0.20
RELATION_LATENT_REGULARIZATION_WEIGHT = 1e-4
RELATION_SURPRISE_MODE = "absolute"
RELATION_POSITIVE_SURPRISE_WEIGHT = 0.50
RELATION_ABSOLUTE_SURPRISE_WEIGHT = 0.50
RELATION_OBJECTIVE_HYBRID = "hybrid_reconstruction"
RELATION_OBJECTIVE_CAUSAL_FUTURE = "causal_future"
RELATION_OBJECTIVE_ONE_CLASS_PROTOTYPE = "one_class_prototype"
RELATION_OBJECTIVE_MODES = {
    RELATION_OBJECTIVE_HYBRID,
    RELATION_OBJECTIVE_CAUSAL_FUTURE,
    RELATION_OBJECTIVE_ONE_CLASS_PROTOTYPE,
}

# The final method owns the short name. Variants carry explicit ablation
# suffixes so figures and tables keep the reader's attention on the full model.
RELATION_MODEL_NAME = "Temporal_Immune_Localizer"
RELATION_GCN_ABLATION_NAME = (
    "Temporal_Immune_Localizer_without_Immune_Encoder"
)
RELATION_GRAPH_FREE_ABLATION_NAME = (
    "Temporal_Immune_Localizer_without_Graph_Message_Passing"
)
RELATION_EDGE_MEMORY_ABLATION_NAME = (
    "TRAIL_without_Edge_Memory"
)
RELATION_NODE_MEMORY_ABLATION_NAME = (
    "TRAIL_without_Node_Memory"
)
RELATION_TEMPORAL_MEMORY_ABLATION_NAME = (
    "TRAIL_without_Temporal_Memory"
)
LEGACY_RELATION_MODEL_NAME = "Temporal_Relation_Localizer"

MODEL_DISPLAY_NAMES = {
    "Vanilla_MLP-AE": "MLP-AE",
    "GCN-AE": "GCN-AE",
    "Immune_GAE": "TRAIL w/o temporal memory",
    "Temporal_Immune_GAE": (
        "TRAIL w/o masked relation objective"
    ),
    RELATION_GCN_ABLATION_NAME: (
        "TRAIL w/o edge-aware encoder"
    ),
    RELATION_GRAPH_FREE_ABLATION_NAME: (
        "TRAIL w/o graph message passing"
    ),
    RELATION_EDGE_MEMORY_ABLATION_NAME: (
        "TRAIL w/o edge memory"
    ),
    RELATION_NODE_MEMORY_ABLATION_NAME: (
        "TRAIL w/o node memory"
    ),
    RELATION_TEMPORAL_MEMORY_ABLATION_NAME: (
        "TRAIL w/o temporal memory"
    ),
    RELATION_MODEL_NAME: "TRAIL",
    LEGACY_RELATION_MODEL_NAME: "TRAIL",
}

EARLY_STOPPING_PATIENCE = 20
MIN_EPOCHS_BEFORE_STOP = 35
RELATION_EARLY_STOPPING_PATIENCE = 20
RELATION_MIN_EPOCHS_BEFORE_STOP = 100
MIN_IMPROVEMENT = 1e-4

HIDDEN_CHANNELS = 64
LATENT_CHANNELS = 32
DROPOUT = 0.10

NEGATIVE_RATIO = 1.0

# Shared objective weights for all models.
TOPOLOGY_WEIGHT = 1.0
EDGE_ATTRIBUTE_WEIGHT = 2.0
NODE_RECONSTRUCTION_WEIGHT = 0.20
LATENT_REGULARIZATION_WEIGHT = 1e-4

GRADIENT_CLIP_NORM = 5.0

PRINT_EVERY = 5
NUM_WORKERS = 0


# ============================================================
# Model configuration
# ============================================================

MODEL_NAMES = [
    "Vanilla_MLP-AE",
    "GCN-AE",
    "Immune_GAE",
]


@dataclass
class ModelConfig:
    model_name: str
    in_channels: int
    edge_attr_dim: int
    hidden_channels: int = HIDDEN_CHANNELS
    latent_channels: int = LATENT_CHANNELS
    dropout: float = DROPOUT


def canonical_model_name(model_name: str) -> str:
    if model_name == LEGACY_RELATION_MODEL_NAME:
        return RELATION_MODEL_NAME
    return model_name


def display_model_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(
        model_name,
        MODEL_DISPLAY_NAMES.get(
            canonical_model_name(model_name),
            model_name.replace("_", " "),
        ),
    )


def is_relation_localizer_name(model_name: str) -> bool:
    return model_name in {
        RELATION_MODEL_NAME,
        RELATION_GCN_ABLATION_NAME,
        RELATION_GRAPH_FREE_ABLATION_NAME,
        RELATION_EDGE_MEMORY_ABLATION_NAME,
        RELATION_NODE_MEMORY_ABLATION_NAME,
        RELATION_TEMPORAL_MEMORY_ABLATION_NAME,
        LEGACY_RELATION_MODEL_NAME,
    }


def relation_mask_fraction(
    epoch: int,
    total_epochs: int,
) -> float:
    del total_epochs
    curriculum_epochs = max(
        RELATION_MASK_CURRICULUM_EPOCHS,
        2,
    )

    progress = np.clip(
        (epoch - 1) / (curriculum_epochs - 1),
        0.0,
        1.0,
    )

    # Smoothly expose the model to the deployment condition instead of
    # jumping from 35% masked edges in training to 100% at inference.
    smooth_progress = progress * progress * (3.0 - 2.0 * progress)

    return float(
        RELATION_MASK_START_FRACTION
        + (
            RELATION_MASK_END_FRACTION
            - RELATION_MASK_START_FRACTION
        )
        * smooth_progress
    )


# ============================================================
# Generic utilities
# ============================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(mode: str) -> torch.device:
    mode = mode.lower()

    if mode == "cpu":
        return torch.device("cpu")

    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE_MODE='cuda'，但 CUDA 不可用。")
        return torch.device("cuda")

    if mode == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("DEVICE_MODE='mps'，但 MPS 不可用。")
        return torch.device("mps")

    if mode == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    raise ValueError(f"未知 DEVICE_MODE：{mode}")


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def configure_dataset_profile(
    args: argparse.Namespace,
) -> None:
    global DATASET_PATH
    global DATASET_TAG
    global DATASET_LABEL
    global EXPECTED_RELATION_SCHEMA_VERSION
    global CHECKPOINT_DIR
    global RESULT_DIR
    global SUMMARY_PATH
    global COMBINED_HISTORY_PATH

    profile = DATASET_PROFILES[args.dataset_profile]

    DATASET_PATH = (
        args.dataset_path.expanduser().resolve()
        if args.dataset_path is not None
        else Path(profile["dataset_path"]).resolve()
    )
    DATASET_TAG = str(profile["dataset_tag"])
    DATASET_LABEL = str(profile["dataset_label"])
    EXPECTED_RELATION_SCHEMA_VERSION = (
        str(args.expected_relation_schema)
        if args.expected_relation_schema is not None
        else str(profile["expected_schema"])
    )
    CHECKPOINT_DIR = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir is not None
        else Path(profile["checkpoint_dir"]).resolve()
    )
    RESULT_DIR = (
        args.result_dir.expanduser().resolve()
        if args.result_dir is not None
        else Path(profile["result_dir"]).resolve()
    )
    SUMMARY_PATH = RESULT_DIR / "training_summary.json"
    COMBINED_HISTORY_PATH = RESULT_DIR / "training_history.csv"


def load_dataset() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"找不到最终数据集：{DATASET_PATH}\n"
            "US-101 请先运行 01e；InterHub 请先运行 05a、05b、05c；"
            "strong-interaction 请先运行 07a、07b、07c。"
        )

    package = torch.load(
        DATASET_PATH,
        weights_only=False,
    )

    if "edge_normalization" not in package:
        raise ValueError(
            "检测到旧版三通道 relation 数据集；"
            "新版 TRAIL 需要重新运行 01b、01d、01e。"
        )

    required = [
        "train_normal",
        "val_normal",
        "node_normalization",
        "edge_normalization",
        "meta",
    ]

    missing = [
        key for key in required
        if key not in package
    ]

    if missing:
        raise KeyError(
            f"最终数据集缺少字段：{missing}"
        )

    if len(package["train_normal"]) == 0:
        raise ValueError("train_normal 为空。")

    if len(package["val_normal"]) == 0:
        raise ValueError("val_normal 为空。")

    observed_schema = package["meta"].get(
        "relation_schema_version"
    )
    if observed_schema != EXPECTED_RELATION_SCHEMA_VERSION:
        raise ValueError(
            "数据集 relation schema 不兼容："
            f"expected={EXPECTED_RELATION_SCHEMA_VERSION}, "
            f"observed={observed_schema!r}。"
            "请检查 --dataset-profile 或 --expected-relation-schema。"
        )

    return package


def scalar_int(value: Any, default: int = -1) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return int(value.detach().cpu().view(-1)[0].item())

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def graph_sequence_key(graph: Any) -> tuple[int, int]:
    sequence_index = scalar_int(
        getattr(graph, "sequence_index", None),
        default=-1,
    )

    if sequence_index >= 0:
        return sequence_index, sequence_index

    return (
        scalar_int(getattr(graph, "global_time", None), default=0),
        scalar_int(getattr(graph, "frame_id", None), default=0),
    )


def split_sequence_by_time_gaps(graphs: list[Any]) -> list[list[Any]]:
    if not graphs:
        return []

    sorted_graphs = sorted(graphs, key=graph_sequence_key)
    chunks: list[list[Any]] = [[sorted_graphs[0]]]
    previous_time = scalar_int(getattr(sorted_graphs[0], "global_time", None), default=0)
    previous_frame = scalar_int(getattr(sorted_graphs[0], "frame_id", None), default=0)
    previous_index = scalar_int(getattr(sorted_graphs[0], "sequence_index", None), default=-1)

    for graph in sorted_graphs[1:]:
        current_time = scalar_int(getattr(graph, "global_time", None), default=0)
        current_frame = scalar_int(getattr(graph, "frame_id", None), default=0)
        current_index = scalar_int(getattr(graph, "sequence_index", None), default=-1)

        is_continuous = True

        if previous_index >= 0 and current_index >= 0 and current_index - previous_index > 1:
            is_continuous = False
        if previous_time and current_time and current_time - previous_time > TEMPORAL_MAX_TIME_GAP_MS:
            is_continuous = False
        if previous_frame and current_frame and current_frame - previous_frame > 3:
            is_continuous = False

        if is_continuous:
            chunks[-1].append(graph)
        else:
            chunks.append([graph])

        previous_time = current_time
        previous_frame = current_frame
        previous_index = current_index

    return chunks


def make_temporal_windows(sequence: list[Any]) -> list[list[Any]]:
    if len(sequence) < TEMPORAL_SEQUENCE_LENGTH:
        return []

    windows = []

    for start in range(
        0,
        len(sequence) - TEMPORAL_SEQUENCE_LENGTH + 1,
        TEMPORAL_SEQUENCE_STRIDE,
    ):
        windows.append(sequence[start : start + TEMPORAL_SEQUENCE_LENGTH])

    return windows


def temporal_training_units_from_chunk(
    chunk: list[Any],
) -> list[list[Any]]:
    if len(chunk) < TEMPORAL_SEQUENCE_LENGTH:
        return []

    if TEMPORAL_TRAIN_ON_FULL_CLIPS:
        return [chunk]

    return make_temporal_windows(chunk)


def build_temporal_training_sequences(
    package: dict[str, Any],
) -> tuple[list[list[Any]], list[list[Any]], dict[str, Any]]:
    paired_cases = package.get("paired_cases", [])
    split_statistics = package.get("meta", {}).get("split_statistics", {})

    train_periods = {
        period
        for period, statistics in split_statistics.items()
        if statistics.get("split") == "train"
    }

    validation_periods = {
        period
        for period, statistics in split_statistics.items()
        if statistics.get("split") == "validation"
    }

    train_sequences: list[list[Any]] = []
    val_sequences: list[list[Any]] = []
    source_case_counts = defaultdict(int)

    for case in paired_cases:
        period = str(case.get("period"))
        normal_graphs = case.get("normal_graphs", [])

        target_sequences = (
            train_sequences
            if period in train_periods
            else val_sequences
            if period in validation_periods
            else None
        )

        if target_sequences is None:
            continue

        source_case_counts[period] += 1

        for chunk in split_sequence_by_time_gaps(normal_graphs):
            target_sequences.extend(
                temporal_training_units_from_chunk(chunk)
            )

    if not train_sequences:
        raise RuntimeError(
            "无法从 train periods 的 paired normal sequences 构建 temporal training windows。"
        )

    if not val_sequences:
        raise RuntimeError(
            "无法从 validation periods 的 paired normal sequences 构建 temporal validation windows。"
        )

    summary = {
        "sequence_source": "paired_cases.normal_graphs",
        "train_periods": sorted(train_periods),
        "validation_periods": sorted(validation_periods),
        "source_case_counts": dict(source_case_counts),
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(val_sequences),
        "minimum_sequence_length": TEMPORAL_SEQUENCE_LENGTH,
        "sequence_stride": TEMPORAL_SEQUENCE_STRIDE,
        "train_on_full_clips": TEMPORAL_TRAIN_ON_FULL_CLIPS,
    }

    return train_sequences, val_sequences, summary


# ============================================================
# Negative edge sampling within each batched graph
# ============================================================

def sample_negative_edges(
    edge_index: torch.Tensor,
    ptr: torch.Tensor,
    negative_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Sample directed non-edge pairs separately inside each graph in a PyG batch.

    Cross-graph node pairs are never sampled.
    """
    edge_index_cpu = edge_index.detach().cpu()
    ptr_cpu = ptr.detach().cpu()

    negative_sources: list[int] = []
    negative_targets: list[int] = []

    graph_count = ptr_cpu.numel() - 1

    for graph_index in range(graph_count):
        start = int(ptr_cpu[graph_index].item())
        end = int(ptr_cpu[graph_index + 1].item())
        node_count = end - start

        if node_count < 2:
            continue

        local_positive_mask = (
            (edge_index_cpu[0] >= start)
            & (edge_index_cpu[0] < end)
        )

        local_edges = (
            edge_index_cpu[:, local_positive_mask]
            - start
        )

        positive_pairs = {
            (
                int(local_edges[0, edge].item()),
                int(local_edges[1, edge].item()),
            )
            for edge in range(local_edges.size(1))
        }

        candidates = [
            (source, target)
            for source in range(node_count)
            for target in range(node_count)
            if (
                source != target
                and (source, target)
                not in positive_pairs
            )
        ]

        if not candidates:
            continue

        requested = max(
            1,
            int(
                round(
                    local_edges.size(1)
                    * negative_ratio
                )
            ),
        )

        sample_count = min(
            requested,
            len(candidates),
        )

        permutation = torch.randperm(
            len(candidates),
            generator=generator,
        )[:sample_count]

        for candidate_index in permutation.tolist():
            source, target = candidates[
                candidate_index
            ]

            negative_sources.append(
                source + start
            )

            negative_targets.append(
                target + start
            )

    if not negative_sources:
        return torch.empty(
            (2, 0),
            dtype=torch.long,
            device=edge_index.device,
        )

    return torch.tensor(
        [
            negative_sources,
            negative_targets,
        ],
        dtype=torch.long,
        device=edge_index.device,
    )


# ============================================================
# Shared decoders
# ============================================================

class PairDecoderInput(nn.Module):
    """
    Construct symmetric and directional pair features.

    Concatenating source, target, absolute difference, and elementwise product
    gives the decoder enough capacity to reconstruct directed interactions while
    keeping the comparison architecture identical for all models.
    """

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        source = edge_index[0]
        target = edge_index[1]

        z_source = z[source]
        z_target = z[target]

        return torch.cat(
            [
                z_source,
                z_target,
                torch.abs(
                    z_source - z_target
                ),
                z_source * z_target,
            ],
            dim=-1,
        )


class TopologyDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        pair_channels = latent_channels * 4

        self.pair_features = PairDecoderInput()

        self.network = nn.Sequential(
            nn.Linear(
                pair_channels,
                hidden_channels,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_channels,
                hidden_channels // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_channels // 2,
                1,
            ),
        )

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.empty(
                0,
                device=z.device,
                dtype=z.dtype,
            )

        pair_features = self.pair_features(
            z,
            edge_index,
        )

        return self.network(
            pair_features
        ).squeeze(-1)


class EdgeAttributeDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int,
        edge_attr_dim: int,
    ) -> None:
        super().__init__()

        pair_channels = latent_channels * 4

        self.pair_features = PairDecoderInput()

        self.network = nn.Sequential(
            nn.Linear(
                pair_channels,
                hidden_channels,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_channels,
                hidden_channels // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_channels // 2,
                edge_attr_dim,
            ),
        )

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.empty(
                (0, self.network[-1].out_features),
                device=z.device,
                dtype=z.dtype,
            )

        pair_features = self.pair_features(
            z,
            edge_index,
        )

        return self.network(
            pair_features
        )


# ============================================================
# Encoders and complete models
# ============================================================

class BaseGraphAutoencoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.topology_decoder = TopologyDecoder(
            latent_channels=config.latent_channels,
            hidden_channels=config.hidden_channels,
        )

        self.edge_attr_decoder = EdgeAttributeDecoder(
            latent_channels=config.latent_channels,
            hidden_channels=config.hidden_channels,
            edge_attr_dim=config.edge_attr_dim,
        )

        self.node_decoder = nn.Sequential(
            nn.Linear(
                config.latent_channels,
                config.hidden_channels,
            ),
            nn.ReLU(),
            nn.Linear(
                config.hidden_channels,
                config.in_channels,
            ),
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def decode_topology(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.topology_decoder(
            z,
            edge_index,
        )

    def decode_edge_attr(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_attr_decoder(
            z,
            edge_index,
        )

    def decode_nodes(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return self.node_decoder(z)


class VanillaMLPAE(BaseGraphAutoencoder):
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__(config)

        self.encoder = nn.Sequential(
            nn.Linear(
                config.in_channels,
                config.hidden_channels,
            ),
            nn.LayerNorm(
                config.hidden_channels
            ),
            nn.ReLU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.hidden_channels,
                config.hidden_channels,
            ),
            nn.LayerNorm(
                config.hidden_channels
            ),
            nn.ReLU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.hidden_channels,
                config.latent_channels,
            ),
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        del edge_index, edge_attr
        return self.encoder(x)


class GCNAE(BaseGraphAutoencoder):
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__(config)

        self.input_projection = nn.Linear(
            config.in_channels,
            config.hidden_channels,
        )

        self.conv1 = GCNConv(
            config.hidden_channels,
            config.hidden_channels,
        )

        self.conv2 = GCNConv(
            config.hidden_channels,
            config.latent_channels,
        )

        self.norm1 = nn.LayerNorm(
            config.hidden_channels
        )

        self.norm2 = nn.LayerNorm(
            config.latent_channels
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        del edge_attr

        hidden = self.input_projection(x)
        hidden = self.conv1(
            hidden,
            edge_index,
        )
        hidden = self.norm1(hidden)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)

        latent = self.conv2(
            hidden,
            edge_index,
        )
        latent = self.norm2(latent)

        return latent


class ImmuneGAE(BaseGraphAutoencoder):
    """
    Edge-aware attention autoencoder.

    The encoder consumes normalized observable directed-relation channels.
    Interaction itself is learned temporally rather than supplied as a label.

    A learned edge gate emphasizes relationships that are important for the
    current message while retaining the original physical edge attributes.
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__(config)

        self.input_projection = nn.Linear(
            config.in_channels,
            config.hidden_channels,
        )

        self.edge_projection = nn.Sequential(
            nn.Linear(
                config.edge_attr_dim,
                config.hidden_channels // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                config.hidden_channels // 2,
                config.edge_attr_dim,
            ),
            nn.Sigmoid(),
        )

        self.gat1 = GATv2Conv(
            in_channels=config.hidden_channels,
            out_channels=config.hidden_channels // 4,
            heads=4,
            concat=True,
            dropout=config.dropout,
            edge_dim=config.edge_attr_dim,
            add_self_loops=True,
        )

        self.gat2 = GATv2Conv(
            in_channels=config.hidden_channels,
            out_channels=config.latent_channels,
            heads=1,
            concat=False,
            dropout=config.dropout,
            edge_dim=config.edge_attr_dim,
            add_self_loops=True,
        )

        self.residual = nn.Linear(
            config.hidden_channels,
            config.latent_channels,
        )

        self.norm1 = nn.LayerNorm(
            config.hidden_channels
        )

        self.norm2 = nn.LayerNorm(
            config.latent_channels
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(x)

        # Gate the physical relation attributes rather than replacing them.
        edge_gate = self.edge_projection(
            edge_attr
        )

        gated_edge_attr = (
            edge_attr
            * (0.5 + edge_gate)
        )

        message = self.gat1(
            hidden,
            edge_index,
            gated_edge_attr,
        )

        message = self.norm1(message)
        message = F.elu(message)
        message = self.dropout(message)

        latent = self.gat2(
            message,
            edge_index,
            gated_edge_attr,
        )

        latent = (
            latent
            + self.residual(hidden)
        )

        latent = self.norm2(latent)

        return latent


class TemporalImmuneGAE(BaseGraphAutoencoder):
    """
    Edge-aware graph autoencoder with vehicle-level temporal memory.

    Each frame is first encoded by the Immune-GAE message-passing encoder. A
    GRUCell then updates a latent memory keyed by persistent vehicle IDs across
    the sequence, so reconstruction uses both current graph context and recent
    vehicle interaction history.
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__(config)

        self.input_projection = nn.Linear(
            config.in_channels,
            config.hidden_channels,
        )

        self.edge_projection = nn.Sequential(
            nn.Linear(
                config.edge_attr_dim,
                config.hidden_channels // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                config.hidden_channels // 2,
                config.edge_attr_dim,
            ),
            nn.Sigmoid(),
        )

        self.gat1 = GATv2Conv(
            in_channels=config.hidden_channels,
            out_channels=config.hidden_channels // 4,
            heads=4,
            concat=True,
            dropout=config.dropout,
            edge_dim=config.edge_attr_dim,
            add_self_loops=True,
        )

        self.gat2 = GATv2Conv(
            in_channels=config.hidden_channels,
            out_channels=config.latent_channels,
            heads=1,
            concat=False,
            dropout=config.dropout,
            edge_dim=config.edge_attr_dim,
            add_self_loops=True,
        )

        self.residual = nn.Linear(
            config.hidden_channels,
            config.latent_channels,
        )

        self.norm1 = nn.LayerNorm(
            config.hidden_channels
        )

        self.norm2 = nn.LayerNorm(
            config.latent_channels
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.temporal_gru = nn.GRUCell(
            config.latent_channels,
            config.latent_channels,
        )

        self.temporal_norm = nn.LayerNorm(
            config.latent_channels
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_frame_raw(
            x,
            edge_index,
            edge_attr,
        )

    def encode_frame_raw(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(x)

        edge_gate = self.edge_projection(
            edge_attr
        )

        gated_edge_attr = (
            edge_attr
            * (0.5 + edge_gate)
        )

        message = self.gat1(
            hidden,
            edge_index,
            gated_edge_attr,
        )

        message = self.norm1(message)
        message = F.elu(message)
        message = self.dropout(message)

        latent = self.gat2(
            message,
            edge_index,
            gated_edge_attr,
        )

        latent = (
            latent
            + self.residual(hidden)
        )

        return self.norm2(latent)

    def encode_sequence(
        self,
        sequence: list[Any],
        device: torch.device,
    ) -> list[torch.Tensor]:
        memory: dict[int, torch.Tensor] = {}
        temporal_embeddings: list[torch.Tensor] = []

        for graph in sequence:
            graph = graph.to(device)
            z_raw = self.encode_frame_raw(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
            )

            if hasattr(graph, "vehicle_ids"):
                vehicle_ids = (
                    graph.vehicle_ids
                    .detach()
                    .cpu()
                    .view(-1)
                    .tolist()
                )
            else:
                vehicle_ids = list(
                    range(z_raw.size(0))
                )

            updated_rows: list[torch.Tensor] = []

            for node_index, vehicle_id in enumerate(vehicle_ids):
                vehicle_id_int = int(vehicle_id)
                current = z_raw[node_index]
                previous = memory.get(vehicle_id_int)

                if previous is None:
                    updated = current
                else:
                    updated = self.temporal_gru(
                        current.unsqueeze(0),
                        previous.unsqueeze(0),
                    ).squeeze(0)

                updated = self.temporal_norm(updated)
                memory[vehicle_id_int] = updated
                updated_rows.append(updated)

            temporal_embeddings.append(
                torch.stack(updated_rows, dim=0)
            )

        return temporal_embeddings


class TemporalImmuneLocalizer(BaseGraphAutoencoder):
    """
    Edge-localization-oriented temporal model.

    The encoder is still Immune/GATv2 based, but training hides a subset of
    current edge attributes and asks the model to predict the normal relation
    from node state, graph context, and temporal vehicle memory.
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__(config)

        self.use_immune_encoder = (
            config.model_name
            not in {
                RELATION_GCN_ABLATION_NAME,
                RELATION_GRAPH_FREE_ABLATION_NAME,
            }
        )
        self.use_graph_message_passing = (
            config.model_name
            != RELATION_GRAPH_FREE_ABLATION_NAME
        )
        self.use_edge_memory = (
            config.model_name
            not in {
                RELATION_EDGE_MEMORY_ABLATION_NAME,
                RELATION_TEMPORAL_MEMORY_ABLATION_NAME,
            }
        )
        self.use_node_memory = (
            config.model_name
            not in {
                RELATION_NODE_MEMORY_ABLATION_NAME,
                RELATION_TEMPORAL_MEMORY_ABLATION_NAME,
            }
        )

        self.input_projection = nn.Linear(
            config.in_channels,
            config.hidden_channels,
        )

        self.edge_mask_token = nn.Parameter(
            torch.zeros(config.edge_attr_dim)
        )

        if self.use_immune_encoder:
            self.edge_projection = nn.Sequential(
                nn.Linear(
                    config.edge_attr_dim,
                    config.hidden_channels // 2,
                ),
                nn.ReLU(),
                nn.Linear(
                    config.hidden_channels // 2,
                    config.edge_attr_dim,
                ),
                nn.Sigmoid(),
            )

            self.gat1 = GATv2Conv(
                in_channels=config.hidden_channels,
                out_channels=config.hidden_channels // 4,
                heads=4,
                concat=True,
                dropout=config.dropout,
                edge_dim=config.edge_attr_dim,
                add_self_loops=True,
            )

            self.gat2 = GATv2Conv(
                in_channels=config.hidden_channels,
                out_channels=config.latent_channels,
                heads=1,
                concat=False,
                dropout=config.dropout,
                edge_dim=config.edge_attr_dim,
                add_self_loops=True,
            )
        elif self.use_graph_message_passing:
            # Ablation: keep the temporal memory and relation-prediction
            # objective, but remove all edge-attribute-aware attention.
            self.gcn1 = GCNConv(
                config.hidden_channels,
                config.hidden_channels,
            )

            self.gcn2 = GCNConv(
                config.hidden_channels,
                config.latent_channels,
            )
        else:
            # Graph-free ablation: every road user is encoded independently.
            # edge_index remains only as a deterministic pair-enumeration
            # device for the shared relation decoder; it never participates in
            # node message passing.
            self.node_mlp1 = nn.Linear(
                config.hidden_channels,
                config.hidden_channels,
            )
            self.node_mlp2 = nn.Linear(
                config.hidden_channels,
                config.latent_channels,
            )

        self.residual = nn.Linear(
            config.hidden_channels,
            config.latent_channels,
        )

        self.norm1 = nn.LayerNorm(
            config.hidden_channels
        )

        self.norm2 = nn.LayerNorm(
            config.latent_channels
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.temporal_gru = nn.GRUCell(
            config.latent_channels,
            config.latent_channels,
        )

        self.temporal_norm = nn.LayerNorm(
            config.latent_channels
        )

        pair_channels = config.latent_channels * 4
        relation_channels = (
            pair_channels
            + config.latent_channels
        )

        self.relation_pair_features = PairDecoderInput()

        self.edge_temporal_gru = nn.GRUCell(
            pair_channels + config.edge_attr_dim,
            config.latent_channels,
        )
        self.edge_temporal_norm = nn.LayerNorm(
            config.latent_channels
        )

        self.relation_predictor = nn.Sequential(
            nn.Linear(
                relation_channels,
                config.hidden_channels,
            ),
            nn.ReLU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.hidden_channels,
                config.hidden_channels // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                config.hidden_channels // 2,
                config.edge_attr_dim,
            ),
        )

        self.next_relation_predictor = nn.Sequential(
            nn.Linear(
                relation_channels,
                config.hidden_channels,
            ),
            nn.ReLU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.hidden_channels,
                config.hidden_channels // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                config.hidden_channels // 2,
                config.edge_attr_dim,
            ),
        )

        # A memoryless reference is trained on the same normal targets.  Its
        # error reduction relative to the temporal decoder is a learned,
        # auditable predictive-influence score; it is not an input label.
        self.memoryless_relation_predictor = nn.Sequential(
            nn.Linear(
                pair_channels,
                config.hidden_channels,
            ),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.hidden_channels,
                config.hidden_channels // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                config.hidden_channels // 2,
                config.edge_attr_dim,
            ),
        )

        self.memoryless_next_relation_predictor = nn.Sequential(
            nn.Linear(
                pair_channels,
                config.hidden_channels,
            ),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.hidden_channels,
                config.hidden_channels // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                config.hidden_channels // 2,
                config.edge_attr_dim,
            ),
        )

        channel_weights = torch.ones(
            config.edge_attr_dim,
            dtype=torch.float32,
        )

        self.register_buffer(
            "relation_channel_weights",
            channel_weights,
        )

        self.register_buffer(
            "safe_relation_prototype_center",
            torch.zeros(config.latent_channels, dtype=torch.float32),
        )
        self.register_buffer(
            "safe_relation_prototype_scale",
            torch.ones(config.latent_channels, dtype=torch.float32),
        )
        self.register_buffer(
            "safe_relation_prototype_initialized",
            torch.tensor(False, dtype=torch.bool),
        )

    @torch.no_grad()
    def update_safe_relation_prototype(
        self,
        transition_states: torch.Tensor,
    ) -> None:
        if transition_states.numel() == 0:
            return
        states = transition_states.detach()
        batch_center = states.mean(dim=0)
        batch_scale = states.std(
            dim=0,
            unbiased=False,
        ).clamp_min(RELATION_PROTOTYPE_MIN_SCALE)
        if not bool(self.safe_relation_prototype_initialized.item()):
            self.safe_relation_prototype_center.copy_(batch_center)
            self.safe_relation_prototype_scale.copy_(batch_scale)
            self.safe_relation_prototype_initialized.fill_(True)
            return
        momentum = RELATION_PROTOTYPE_MOMENTUM
        self.safe_relation_prototype_center.mul_(momentum).add_(
            batch_center,
            alpha=1.0 - momentum,
        )
        self.safe_relation_prototype_scale.mul_(momentum).add_(
            batch_scale,
            alpha=1.0 - momentum,
        )

    def safe_relation_prototype_distance(
        self,
        transition_states: torch.Tensor,
    ) -> torch.Tensor:
        if transition_states.numel() == 0:
            return torch.empty(
                0,
                device=transition_states.device,
                dtype=transition_states.dtype,
            )
        if not bool(self.safe_relation_prototype_initialized.item()):
            return torch.zeros(
                transition_states.size(0),
                device=transition_states.device,
                dtype=transition_states.dtype,
            )
        center = self.safe_relation_prototype_center.detach().to(
            transition_states
        )
        scale = self.safe_relation_prototype_scale.detach().to(
            transition_states
        ).clamp_min(RELATION_PROTOTYPE_MIN_SCALE)
        standardized = (transition_states - center) / scale
        return torch.sqrt(standardized.pow(2).mean(dim=1) + 1e-8)

    def apply_edge_mask(
        self,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if edge_mask is None or edge_attr.numel() == 0:
            return edge_attr

        mask = edge_mask.to(
            device=edge_attr.device,
            dtype=edge_attr.dtype,
        ).unsqueeze(-1)

        mask_token = self.edge_mask_token.view(
            1,
            -1,
        ).expand_as(edge_attr)

        return (
            edge_attr
            * (1.0 - mask)
            + mask_token
            * mask
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_frame_raw(
            x,
            edge_index,
            edge_attr,
        )

    def encode_frame_raw(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.input_projection(x)

        if not self.use_graph_message_passing:
            message = self.node_mlp1(hidden)
            message = self.norm1(message)
            message = F.relu(message)
            message = self.dropout(message)
            latent = self.node_mlp2(message)
            latent = latent + self.residual(hidden)
            return self.norm2(latent)

        if not self.use_immune_encoder:
            message = self.gcn1(
                hidden,
                edge_index,
            )

            message = self.norm1(message)
            message = F.relu(message)
            message = self.dropout(message)

            latent = self.gcn2(
                message,
                edge_index,
            )

            latent = (
                latent
                + self.residual(hidden)
            )

            return self.norm2(latent)

        masked_edge_attr = self.apply_edge_mask(
            edge_attr,
            edge_mask,
        )

        edge_gate = self.edge_projection(
            masked_edge_attr
        )

        gated_edge_attr = (
            masked_edge_attr
            * (0.5 + edge_gate)
        )

        message = self.gat1(
            hidden,
            edge_index,
            gated_edge_attr,
        )

        message = self.norm1(message)
        message = F.elu(message)
        message = self.dropout(message)

        latent = self.gat2(
            message,
            edge_index,
            gated_edge_attr,
        )

        latent = (
            latent
            + self.residual(hidden)
        )

        return self.norm2(latent)

    def encode_sequence(
        self,
        sequence: list[Any],
        device: torch.device,
        edge_masks: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        memory: dict[int, torch.Tensor] = {}
        temporal_embeddings: list[torch.Tensor] = []

        for frame_index, graph in enumerate(sequence):
            graph = graph.to(device)

            edge_mask = (
                None
                if edge_masks is None
                else edge_masks[frame_index].to(device)
            )

            z_raw = self.encode_frame_raw(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                edge_mask=edge_mask,
            )

            if hasattr(graph, "vehicle_ids"):
                vehicle_ids = (
                    graph.vehicle_ids
                    .detach()
                    .cpu()
                    .view(-1)
                    .tolist()
                )
            else:
                vehicle_ids = list(
                    range(z_raw.size(0))
                )

            updated_rows: list[torch.Tensor] = []

            for node_index, vehicle_id in enumerate(vehicle_ids):
                vehicle_id_int = int(vehicle_id)
                current = z_raw[node_index]
                previous = memory.get(vehicle_id_int)

                if not self.use_node_memory or previous is None:
                    updated = current
                else:
                    updated = self.temporal_gru(
                        current.unsqueeze(0),
                        previous.unsqueeze(0),
                    ).squeeze(0)

                updated = self.temporal_norm(updated)
                memory[vehicle_id_int] = updated
                updated_rows.append(updated)

            temporal_embeddings.append(
                torch.stack(updated_rows, dim=0)
            )

        return temporal_embeddings

    def relation_features(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.relation_pair_features(
            z,
            edge_index,
        )

    def encode_relation_sequence(
        self,
        sequence: list[Any],
        device: torch.device,
        edge_masks: list[torch.Tensor] | None = None,
        return_updated_states: bool = False,
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor]]
        | tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
        ]
    ):
        """Encode node histories and causal directed-pair histories.

        ``prior_edge_embeddings[t]`` contains the pair state available before
        frame ``t`` is observed.  It is therefore safe for predicting the
        current relation without letting the anomaly update the memory first.

        ``updated_edge_embeddings[t]`` contains the state after frame ``t`` and
        is used for one-step-ahead prediction of frame ``t + 1``.
        """

        node_embeddings = self.encode_sequence(
            sequence,
            device,
            edge_masks=edge_masks,
        )
        pair_memory: dict[
            tuple[int, int],
            torch.Tensor,
        ] = {}
        prior_edge_embeddings: list[torch.Tensor] = []
        updated_edge_embeddings: list[torch.Tensor] = []

        for frame_index, (graph, z) in enumerate(
            zip(sequence, node_embeddings)
        ):
            graph = graph.to(device)
            edge_mask = (
                None
                if edge_masks is None
                else edge_masks[frame_index].to(device)
            )
            masked_observation = self.apply_edge_mask(
                graph.edge_attr,
                edge_mask,
            )
            pair_features = self.relation_features(
                z,
                graph.edge_index,
            )

            if pair_features.numel() == 0:
                empty_state = torch.empty(
                    (0, self.config.latent_channels),
                    device=device,
                    dtype=z.dtype,
                )
                prior_edge_embeddings.append(empty_state)
                updated_edge_embeddings.append(empty_state)
                continue

            edge_inputs = torch.cat(
                [pair_features, masked_observation],
                dim=1,
            )

            if not self.use_edge_memory:
                zero_state = torch.zeros(
                    (
                        pair_features.size(0),
                        self.config.latent_channels,
                    ),
                    device=device,
                    dtype=pair_features.dtype,
                )
                prior_edge_embeddings.append(zero_state)
                updated_edge_embeddings.append(zero_state)
                continue

            keys = directed_vehicle_edge_keys(graph)
            prior_rows: list[torch.Tensor] = []
            updated_rows: list[torch.Tensor] = []

            for edge_position, key in enumerate(keys):
                current = edge_inputs[edge_position]
                previous = pair_memory.get(key)

                if previous is None:
                    previous = torch.zeros(
                        self.config.latent_channels,
                        device=device,
                        dtype=current.dtype,
                    )

                prior_rows.append(previous)

                updated = self.edge_temporal_gru(
                    current.unsqueeze(0),
                    previous.unsqueeze(0),
                ).squeeze(0)
                updated = self.edge_temporal_norm(updated)
                pair_memory[key] = updated
                updated_rows.append(updated)

            prior_edge_embeddings.append(
                torch.stack(prior_rows, dim=0)
            )
            updated_edge_embeddings.append(
                torch.stack(updated_rows, dim=0)
            )

        if return_updated_states:
            return (
                node_embeddings,
                prior_edge_embeddings,
                updated_edge_embeddings,
            )

        return node_embeddings, prior_edge_embeddings

    def relation_decoder_features(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor | None,
    ) -> torch.Tensor:
        pair_features = self.relation_features(
            z,
            edge_index,
        )

        if edge_state is None:
            edge_state = torch.zeros(
                (
                    pair_features.size(0),
                    self.config.latent_channels,
                ),
                device=pair_features.device,
                dtype=pair_features.dtype,
            )

        return torch.cat(
            [pair_features, edge_state],
            dim=1,
        )

    def predict_relation_attr(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.relation_predictor(
            self.relation_decoder_features(
                z,
                edge_index,
                edge_state,
            )
        )

    def predict_next_relation_attr(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.next_relation_predictor(
            self.relation_decoder_features(
                z,
                edge_index,
                edge_state,
            )
        )

    def predict_memoryless_relation_attr(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.memoryless_relation_predictor(
            self.relation_features(
                z,
                edge_index,
            )
        )

    def predict_memoryless_next_relation_attr(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.memoryless_next_relation_predictor(
            self.relation_features(
                z,
                edge_index,
            )
        )

    def decode_edge_attr(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_relation_attr(
            z,
            edge_index,
        )

    def edge_surprise_scores(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_state: torch.Tensor | None = None,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.predict_relation_attr(
            z,
            edge_index,
            edge_state,
        )

        residual = (
            edge_attr
            - expected
        )

        positive_delta = F.relu(
            residual
        )

        absolute_delta = torch.abs(
            residual
        )

        score_mode = (
            RELATION_SURPRISE_MODE
            if mode is None
            else mode
        )

        if score_mode == "positive":
            surprise = positive_delta
        elif score_mode == "absolute":
            surprise = absolute_delta
        elif score_mode == "hybrid":
            surprise = (
                RELATION_POSITIVE_SURPRISE_WEIGHT
                * positive_delta
                + RELATION_ABSOLUTE_SURPRISE_WEIGHT
                * absolute_delta
            )
        else:
            raise ValueError(
                f"Unknown relation surprise mode: {score_mode}"
            )

        weighted_surprise = (
            surprise
            * self.relation_channel_weights.view(1, -1)
        )

        return weighted_surprise.sum(dim=1), expected

    def edge_surprise_components(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        expected = self.predict_relation_attr(
            z,
            edge_index,
            edge_state,
        )
        memoryless_expected = (
            self.predict_memoryless_relation_attr(
                z,
                edge_index,
            )
        )

        residual = edge_attr - expected
        memoryless_residual = (
            edge_attr - memoryless_expected
        )
        channel_weights = (
            self.relation_channel_weights
            .view(1, -1)
        )
        temporal_error = (
            torch.abs(residual)
            * channel_weights
        ).sum(dim=1)
        memoryless_error = (
            torch.abs(memoryless_residual)
            * channel_weights
        ).sum(dim=1)

        return {
            "expected": expected,
            "memoryless_expected": (
                memoryless_expected
            ),
            "residual": residual,
            "positive": (
                F.relu(residual)
                * channel_weights
            ),
            "absolute": (
                torch.abs(residual)
                * channel_weights
            ),
            "predictive_influence": F.relu(
                memoryless_error
                - temporal_error
            ),
        }

    def score_sequence_masked_groups(
        self,
        sequence: list[Any],
        device: torch.device,
        *,
        group_count: int = RELATION_INFERENCE_MASK_GROUPS,
        surprise_mode: str = RELATION_SURPRISE_MODE,
    ) -> list[dict[str, torch.Tensor]]:
        return score_relation_sequence_masked_groups(
            self,
            sequence,
            device,
            group_count=group_count,
            surprise_mode=surprise_mode,
        )


# Compatibility for older imports and checkpoints.
TemporalRelationLocalizer = TemporalImmuneLocalizer


def build_model(
    config: ModelConfig,
) -> BaseGraphAutoencoder:
    if config.model_name == "Vanilla_MLP-AE":
        return VanillaMLPAE(config)

    if config.model_name == "GCN-AE":
        return GCNAE(config)

    if config.model_name == "Immune_GAE":
        return ImmuneGAE(config)

    if config.model_name == "Temporal_Immune_GAE":
        return TemporalImmuneGAE(config)

    if is_relation_localizer_name(
        config.model_name
    ):
        return TemporalImmuneLocalizer(config)

    raise KeyError(
        f"未知模型：{config.model_name}"
    )


# ============================================================
# Loss and metrics
# ============================================================

@dataclass
class LossOutput:
    total: torch.Tensor
    topology: torch.Tensor
    edge_attr: torch.Tensor
    node: torch.Tensor
    latent_reg: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor
    positive_probability: torch.Tensor
    negative_probability: torch.Tensor


def compute_batch_loss(
    model: BaseGraphAutoencoder,
    batch: Any,
    negative_generator: torch.Generator,
) -> LossOutput:
    z = model.encode(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
    )

    negative_edge_index = sample_negative_edges(
        edge_index=batch.edge_index,
        ptr=batch.ptr,
        negative_ratio=NEGATIVE_RATIO,
        generator=negative_generator,
    )

    positive_logits = model.decode_topology(
        z,
        batch.edge_index,
    )

    negative_logits = model.decode_topology(
        z,
        negative_edge_index,
    )

    positive_loss_raw = F.binary_cross_entropy_with_logits(
        positive_logits,
        torch.ones_like(
            positive_logits
        ),
        reduction="none",
    )

    positive_loss = positive_loss_raw.mean()

    if negative_logits.numel() > 0:
        negative_loss = (
            F.binary_cross_entropy_with_logits(
                negative_logits,
                torch.zeros_like(
                    negative_logits
                ),
            )
        )

        negative_probability = torch.sigmoid(
            negative_logits
        ).mean()
    else:
        negative_loss = torch.zeros(
            (),
            device=z.device,
        )

        negative_probability = torch.zeros(
            (),
            device=z.device,
        )

    topology_loss = (
        positive_loss
        + negative_loss
    )

    predicted_edge_attr = (
        model.decode_edge_attr(
            z,
            batch.edge_index,
        )
    )

    edge_attribute_loss = F.smooth_l1_loss(
        predicted_edge_attr,
        batch.edge_attr,
        beta=0.05,
    )

    predicted_nodes = model.decode_nodes(z)

    node_loss = F.smooth_l1_loss(
        predicted_nodes,
        batch.x,
        beta=0.10,
    )

    latent_regularization = (
        z.pow(2).mean()
    )

    total_loss = (
        TOPOLOGY_WEIGHT
        * topology_loss
        + EDGE_ATTRIBUTE_WEIGHT
        * edge_attribute_loss
        + NODE_RECONSTRUCTION_WEIGHT
        * node_loss
        + LATENT_REGULARIZATION_WEIGHT
        * latent_regularization
    )

    positive_probability = torch.sigmoid(
        positive_logits
    ).mean()

    return LossOutput(
        total=total_loss,
        topology=topology_loss,
        edge_attr=edge_attribute_loss,
        node=node_loss,
        latent_reg=latent_regularization,
        positive=positive_loss,
        negative=negative_loss,
        positive_probability=positive_probability,
        negative_probability=negative_probability,
    )


class MetricAccumulator:
    def __init__(self) -> None:
        self.total_weight = 0
        self.values = defaultdict(float)

    def update(
        self,
        loss_output: LossOutput,
        graph_count: int,
    ) -> None:
        self.total_weight += graph_count

        for field_name in [
            "total",
            "topology",
            "edge_attr",
            "node",
            "latent_reg",
            "positive",
            "negative",
            "positive_probability",
            "negative_probability",
        ]:
            value = getattr(
                loss_output,
                field_name,
            )

            self.values[
                field_name
            ] += (
                float(
                    value.detach().cpu().item()
                )
                * graph_count
            )

    def mean(self) -> dict[str, float]:
        denominator = max(
            self.total_weight,
            1,
        )

        return {
            key: value / denominator
            for key, value in self.values.items()
        }


# ============================================================
# Epoch loops
# ============================================================

def train_one_epoch(
    model: BaseGraphAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()

    accumulator = MetricAccumulator()

    negative_generator = torch.Generator(
        device="cpu"
    )

    negative_generator.manual_seed(
        RANDOM_SEED
        + epoch * 10_000
        + 1
    )

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        loss_output = compute_batch_loss(
            model,
            batch,
            negative_generator,
        )

        if not torch.isfinite(
            loss_output.total
        ):
            raise FloatingPointError(
                "训练 loss 出现 NaN/Inf。"
            )

        loss_output.total.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_NORM,
        )

        optimizer.step()

        graph_count = int(
            batch.num_graphs
        )

        accumulator.update(
            loss_output,
            graph_count,
        )

    return accumulator.mean()


@torch.no_grad()
def evaluate_one_epoch(
    model: BaseGraphAutoencoder,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.eval()

    accumulator = MetricAccumulator()

    negative_generator = torch.Generator(
        device="cpu"
    )

    # Fixed-but-epoch-specific validation negatives.
    negative_generator.manual_seed(
        RANDOM_SEED
        + epoch * 10_000
        + 2
    )

    for batch in loader:
        batch = batch.to(device)

        loss_output = compute_batch_loss(
            model,
            batch,
            negative_generator,
        )

        graph_count = int(
            batch.num_graphs
        )

        accumulator.update(
            loss_output,
            graph_count,
        )

    return accumulator.mean()


def single_graph_ptr(graph: Any) -> torch.Tensor:
    return torch.tensor(
        [0, int(graph.num_nodes)],
        dtype=torch.long,
        device=graph.edge_index.device,
    )


def compute_temporal_sequence_loss(
    model: TemporalImmuneGAE,
    sequence: list[Any],
    device: torch.device,
    negative_generator: torch.Generator,
) -> LossOutput:
    device_sequence = [
        graph.clone().to(device)
        for graph in sequence
    ]

    temporal_embeddings = model.encode_sequence(
        device_sequence,
        device,
    )

    frame_losses: list[LossOutput] = []

    for graph, z in zip(
        device_sequence,
        temporal_embeddings,
    ):
        negative_edge_index = sample_negative_edges(
            edge_index=graph.edge_index,
            ptr=single_graph_ptr(graph),
            negative_ratio=NEGATIVE_RATIO,
            generator=negative_generator,
        )

        positive_logits = model.decode_topology(
            z,
            graph.edge_index,
        )

        negative_logits = model.decode_topology(
            z,
            negative_edge_index,
        )

        positive_loss_raw = F.binary_cross_entropy_with_logits(
            positive_logits,
            torch.ones_like(positive_logits),
            reduction="none",
        )

        positive_loss = positive_loss_raw.mean()

        if negative_logits.numel() > 0:
            negative_loss = F.binary_cross_entropy_with_logits(
                negative_logits,
                torch.zeros_like(negative_logits),
            )

            negative_probability = torch.sigmoid(
                negative_logits
            ).mean()
        else:
            negative_loss = torch.zeros(
                (),
                device=z.device,
            )

            negative_probability = torch.zeros(
                (),
                device=z.device,
            )

        topology_loss = (
            positive_loss
            + negative_loss
        )

        predicted_edge_attr = model.decode_edge_attr(
            z,
            graph.edge_index,
        )

        edge_attribute_loss = F.smooth_l1_loss(
            predicted_edge_attr,
            graph.edge_attr,
            beta=0.05,
        )

        predicted_nodes = model.decode_nodes(z)

        node_loss = F.smooth_l1_loss(
            predicted_nodes,
            graph.x,
            beta=0.10,
        )

        latent_regularization = z.pow(2).mean()

        total_loss = (
            TOPOLOGY_WEIGHT
            * topology_loss
            + EDGE_ATTRIBUTE_WEIGHT
            * edge_attribute_loss
            + NODE_RECONSTRUCTION_WEIGHT
            * node_loss
            + LATENT_REGULARIZATION_WEIGHT
            * latent_regularization
        )

        frame_losses.append(
            LossOutput(
                total=total_loss,
                topology=topology_loss,
                edge_attr=edge_attribute_loss,
                node=node_loss,
                latent_reg=latent_regularization,
                positive=positive_loss,
                negative=negative_loss,
                positive_probability=torch.sigmoid(
                    positive_logits
                ).mean(),
                negative_probability=negative_probability,
            )
        )

    return LossOutput(
        total=torch.stack([loss.total for loss in frame_losses]).mean(),
        topology=torch.stack([loss.topology for loss in frame_losses]).mean(),
        edge_attr=torch.stack([loss.edge_attr for loss in frame_losses]).mean(),
        node=torch.stack([loss.node for loss in frame_losses]).mean(),
        latent_reg=torch.stack([loss.latent_reg for loss in frame_losses]).mean(),
        positive=torch.stack([loss.positive for loss in frame_losses]).mean(),
        negative=torch.stack([loss.negative for loss in frame_losses]).mean(),
        positive_probability=torch.stack(
            [loss.positive_probability for loss in frame_losses]
        ).mean(),
        negative_probability=torch.stack(
            [loss.negative_probability for loss in frame_losses]
        ).mean(),
    )


def train_temporal_one_epoch(
    model: TemporalImmuneGAE,
    sequences: list[list[Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()

    accumulator = MetricAccumulator()
    shuffled_sequences = list(sequences)
    random.Random(
        RANDOM_SEED + epoch
    ).shuffle(shuffled_sequences)

    negative_generator = torch.Generator(
        device="cpu"
    )

    negative_generator.manual_seed(
        RANDOM_SEED
        + epoch * 10_000
        + 101
    )

    for sequence in shuffled_sequences:
        optimizer.zero_grad(
            set_to_none=True
        )

        loss_output = compute_temporal_sequence_loss(
            model=model,
            sequence=sequence,
            device=device,
            negative_generator=negative_generator,
        )

        if not torch.isfinite(loss_output.total):
            raise FloatingPointError(
                "Temporal 训练 loss 出现 NaN/Inf。"
            )

        loss_output.total.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_NORM,
        )

        optimizer.step()

        accumulator.update(
            loss_output,
            len(sequence),
        )

    return accumulator.mean()


@torch.no_grad()
def evaluate_temporal_one_epoch(
    model: TemporalImmuneGAE,
    sequences: list[list[Any]],
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.eval()

    accumulator = MetricAccumulator()

    negative_generator = torch.Generator(
        device="cpu"
    )

    negative_generator.manual_seed(
        RANDOM_SEED
        + epoch * 10_000
        + 102
    )

    for sequence in sequences:
        loss_output = compute_temporal_sequence_loss(
            model=model,
            sequence=sequence,
            device=device,
            negative_generator=negative_generator,
        )

        accumulator.update(
            loss_output,
            len(sequence),
        )

    return accumulator.mean()


@dataclass
class RelationLossOutput:
    total: torch.Tensor
    masked_edge: torch.Tensor
    next_edge: torch.Tensor
    memoryless_reference: torch.Tensor
    compactness: torch.Tensor
    prototype_compactness: torch.Tensor
    topology: torch.Tensor
    latent_reg: torch.Tensor
    positive_probability: torch.Tensor
    negative_probability: torch.Tensor
    masked_edge_count: float
    next_edge_count: float
    persistent_pair_count: float


class RelationMetricAccumulator:
    def __init__(self) -> None:
        self.total_weight = 0
        self.values = defaultdict(float)

    def update(
        self,
        loss_output: RelationLossOutput,
        graph_count: int,
    ) -> None:
        self.total_weight += graph_count

        tensor_fields = [
            "total",
            "masked_edge",
            "next_edge",
            "memoryless_reference",
            "compactness",
            "prototype_compactness",
            "topology",
            "latent_reg",
            "positive_probability",
            "negative_probability",
        ]

        scalar_fields = [
            "masked_edge_count",
            "next_edge_count",
            "persistent_pair_count",
        ]

        for field_name in tensor_fields:
            value = getattr(
                loss_output,
                field_name,
            )

            self.values[field_name] += (
                float(value.detach().cpu().item())
                * graph_count
            )

        for field_name in scalar_fields:
            self.values[field_name] += (
                float(getattr(loss_output, field_name))
                * graph_count
            )

    def mean(self) -> dict[str, float]:
        denominator = max(
            self.total_weight,
            1,
        )

        return {
            key: value / denominator
            for key, value in self.values.items()
        }


def random_edge_mask(
    edge_count: int,
    generator: torch.Generator,
    device: torch.device,
    mask_fraction: float,
    full_mask_probability: float = 0.0,
) -> torch.Tensor:
    if edge_count <= 0:
        return torch.empty(
            0,
            dtype=torch.bool,
            device=device,
        )

    use_full_mask = (
        full_mask_probability > 0.0
        and float(
            torch.rand(
                1,
                generator=generator,
            ).item()
        )
        < full_mask_probability
    )

    effective_fraction = (
        1.0
        if use_full_mask
        else float(
            np.clip(
                mask_fraction,
                0.0,
                1.0,
            )
        )
    )

    mask_count = max(
        1,
        int(round(edge_count * effective_fraction)),
    )

    mask_count = min(
        mask_count,
        edge_count,
    )

    permutation = torch.randperm(
        edge_count,
        generator=generator,
    )[:mask_count]

    mask = torch.zeros(
        edge_count,
        dtype=torch.bool,
    )

    mask[permutation] = True

    return mask.to(device)


def directed_vehicle_edge_keys(
    graph: Any,
) -> list[tuple[int, int]]:
    if hasattr(graph, "vehicle_ids"):
        vehicle_ids = (
            graph.vehicle_ids
            .detach()
            .cpu()
            .view(-1)
            .tolist()
        )
    else:
        vehicle_ids = list(
            range(int(graph.num_nodes))
        )

    edge_index = (
        graph.edge_index
        .detach()
        .cpu()
    )

    return [
        (
            int(vehicle_ids[int(source)]),
            int(vehicle_ids[int(target)]),
        )
        for source, target in zip(
            edge_index[0].tolist(),
            edge_index[1].tolist(),
        )
    ]


def relation_edge_group_ids(
    graph: Any,
    group_count: int,
    device: torch.device,
) -> torch.Tensor:
    if group_count <= 0:
        raise ValueError("group_count must be positive.")

    group_ids = []

    for source_vehicle, target_vehicle in (
        directed_vehicle_edge_keys(graph)
    ):
        low, high = sorted(
            (
                int(source_vehicle),
                int(target_vehicle),
            )
        )

        # Stable arithmetic hashing keeps both directed versions of the same
        # physical vehicle pair in one mask group.
        group_ids.append(
            (
                low * 1_000_003
                + high * 9_176
            )
            % group_count
        )

    return torch.tensor(
        group_ids,
        dtype=torch.long,
        device=device,
    )


@torch.no_grad()
def score_relation_sequence_masked_groups(
    model: TemporalImmuneLocalizer,
    sequence: list[Any],
    device: torch.device,
    *,
    group_count: int = RELATION_INFERENCE_MASK_GROUPS,
    surprise_mode: str = RELATION_SURPRISE_MODE,
) -> list[dict[str, torch.Tensor]]:
    """
    Score every edge while hiding the edge being scored.

    Edges are partitioned by undirected vehicle pair. Each pass masks one
    group, so the target relation and its reverse direction are hidden while
    the remaining relations still provide graph context.

    Persistent pairs use causal one-step-ahead surprise: frame ``t`` is scored
    from the prediction made after frame ``t - 1``. Newly appearing pairs fall
    back to masked current-relation surprise.
    """
    device_sequence = [
        graph.clone().to(device)
        for graph in sequence
    ]

    score_buffers = [
        torch.zeros(
            int(graph.edge_index.size(1)),
            device=device,
            dtype=graph.edge_attr.dtype,
        )
        for graph in device_sequence
    ]

    positive_score_buffers = [
        torch.zeros_like(scores)
        for scores in score_buffers
    ]

    absolute_score_buffers = [
        torch.zeros_like(scores)
        for scores in score_buffers
    ]

    influence_score_buffers = [
        torch.zeros_like(scores)
        for scores in score_buffers
    ]

    expected_buffers = [
        torch.zeros_like(graph.edge_attr)
        for graph in device_sequence
    ]

    causal_next_score_buffers = [
        torch.zeros_like(scores)
        for scores in score_buffers
    ]

    causal_next_positive_buffers = [
        torch.zeros_like(scores)
        for scores in score_buffers
    ]

    causal_next_absolute_buffers = [
        torch.zeros_like(scores)
        for scores in score_buffers
    ]

    causal_next_influence_buffers = [
        torch.zeros_like(scores)
        for scores in score_buffers
    ]

    causal_next_expected_buffers = [
        torch.zeros_like(graph.edge_attr)
        for graph in device_sequence
    ]

    causal_next_covered_buffers = [
        torch.zeros(
            int(graph.edge_index.size(1)),
            device=device,
            dtype=torch.bool,
        )
        for graph in device_sequence
    ]

    causal_prototype_score_buffers = [
        torch.zeros_like(scores)
        for scores in score_buffers
    ]

    covered_buffers = [
        torch.zeros(
            int(graph.edge_index.size(1)),
            device=device,
            dtype=torch.bool,
        )
        for graph in device_sequence
    ]

    edge_group_ids = [
        relation_edge_group_ids(
            graph,
            group_count,
            device,
        )
        for graph in device_sequence
    ]

    for group_index in range(group_count):
        edge_masks = [
            group_ids == group_index
            for group_ids in edge_group_ids
        ]

        if not any(
            bool(mask.any().item())
            for mask in edge_masks
        ):
            continue

        (
            temporal_embeddings,
            prior_edge_embeddings,
            updated_edge_embeddings,
        ) = model.encode_relation_sequence(
            device_sequence,
            device,
            edge_masks=edge_masks,
            return_updated_states=True,
        )

        for (
            graph,
            z,
            edge_state,
            edge_mask,
            score_buffer,
            positive_score_buffer,
            absolute_score_buffer,
            influence_score_buffer,
            expected_buffer,
            covered_buffer,
        ) in zip(
            device_sequence,
            temporal_embeddings,
            prior_edge_embeddings,
            edge_masks,
            score_buffers,
            positive_score_buffers,
            absolute_score_buffers,
            influence_score_buffers,
            expected_buffers,
            covered_buffers,
        ):
            if edge_mask.numel() == 0:
                continue

            components = (
                model.edge_surprise_components(
                    z,
                    graph.edge_index,
                    graph.edge_attr,
                    edge_state,
                )
            )

            positive_scores = components[
                "positive"
            ].sum(dim=1)

            absolute_scores = components[
                "absolute"
            ].sum(dim=1)
            influence_scores = components[
                "predictive_influence"
            ]

            if surprise_mode == "positive":
                edge_scores = positive_scores
            elif surprise_mode == "absolute":
                edge_scores = absolute_scores
            elif surprise_mode == "hybrid":
                edge_scores = (
                    RELATION_POSITIVE_SURPRISE_WEIGHT
                    * positive_scores
                    + RELATION_ABSOLUTE_SURPRISE_WEIGHT
                    * absolute_scores
                )
            else:
                raise ValueError(
                    "Unknown relation surprise mode: "
                    f"{surprise_mode}"
                )

            expected = components["expected"]

            score_buffer[edge_mask] = (
                edge_scores[edge_mask]
            )

            positive_score_buffer[edge_mask] = (
                positive_scores[edge_mask]
            )

            absolute_score_buffer[edge_mask] = (
                absolute_scores[edge_mask]
            )

            influence_score_buffer[edge_mask] = (
                influence_scores[edge_mask]
            )

            expected_buffer[edge_mask] = (
                expected[edge_mask]
            )

            covered_buffer[edge_mask] = True

        # For a persistent pair, score frame t from the prediction made at
        # frame t-1.  The predictor uses the pair state after t-1, before any
        # information from frame t can update that state.
        for frame_index in range(
            len(device_sequence) - 1
        ):
            current_graph = device_sequence[frame_index]
            next_graph = device_sequence[frame_index + 1]
            current_z = temporal_embeddings[frame_index]
            current_updated_state = (
                updated_edge_embeddings[frame_index]
            )
            next_updated_state = updated_edge_embeddings[frame_index + 1]

            current_indices, next_indices = (
                matched_persistent_edge_indices(
                    current_graph,
                    next_graph,
                )
            )

            if current_indices.numel() == 0:
                continue

            predicted_next = model.predict_next_relation_attr(
                current_z,
                current_graph.edge_index,
                current_updated_state,
            )
            memoryless_next = (
                model.predict_memoryless_next_relation_attr(
                    current_z,
                    current_graph.edge_index,
                )
            )

            next_target = next_graph.edge_attr[next_indices]
            temporal_residual = (
                next_target
                - predicted_next[current_indices]
            )
            memoryless_residual = (
                next_target
                - memoryless_next[current_indices]
            )
            channel_weights = (
                model.relation_channel_weights
                .view(1, -1)
            )

            temporal_positive = (
                F.relu(temporal_residual)
                * channel_weights
            ).sum(dim=1)
            temporal_absolute = (
                torch.abs(temporal_residual)
                * channel_weights
            ).sum(dim=1)
            temporal_error = temporal_absolute
            memoryless_error = (
                torch.abs(memoryless_residual)
                * channel_weights
            ).sum(dim=1)
            temporal_influence = F.relu(
                memoryless_error
                - temporal_error
            )

            if surprise_mode == "positive":
                temporal_scores = temporal_positive
            elif surprise_mode == "absolute":
                temporal_scores = temporal_absolute
            elif surprise_mode == "hybrid":
                temporal_scores = (
                    RELATION_POSITIVE_SURPRISE_WEIGHT
                    * temporal_positive
                    + RELATION_ABSOLUTE_SURPRISE_WEIGHT
                    * temporal_absolute
                )
            else:
                raise ValueError(
                    "Unknown relation surprise mode: "
                    f"{surprise_mode}"
                )

            next_group_mask = edge_masks[
                frame_index + 1
            ][next_indices]
            if not bool(next_group_mask.any().item()):
                continue

            selected_next_indices = next_indices[
                next_group_mask
            ]
            selected_current_indices = current_indices[
                next_group_mask
            ]
            selected_transition_states = (
                next_updated_state[selected_next_indices]
                - current_updated_state[selected_current_indices]
            )
            prototype_scores = model.safe_relation_prototype_distance(
                selected_transition_states
            )

            causal_next_score_buffers[
                frame_index + 1
            ][selected_next_indices] = temporal_scores[
                next_group_mask
            ]
            causal_next_positive_buffers[
                frame_index + 1
            ][selected_next_indices] = temporal_positive[
                next_group_mask
            ]
            causal_next_absolute_buffers[
                frame_index + 1
            ][selected_next_indices] = temporal_absolute[
                next_group_mask
            ]
            causal_next_influence_buffers[
                frame_index + 1
            ][selected_next_indices] = temporal_influence[
                next_group_mask
            ]
            causal_next_expected_buffers[
                frame_index + 1
            ][selected_next_indices] = predicted_next[
                selected_current_indices
            ]
            causal_next_covered_buffers[
                frame_index + 1
            ][selected_next_indices] = True
            causal_prototype_score_buffers[
                frame_index + 1
            ][selected_next_indices] = prototype_scores

    for covered in covered_buffers:
        if covered.numel() > 0 and not bool(
            covered.all().item()
        ):
            raise RuntimeError(
                "Grouped masked inference did not cover every edge."
            )

    outputs: list[dict[str, torch.Tensor]] = []

    for (
        current_scores,
        current_positive,
        current_absolute,
        current_influence,
        current_expected,
        temporal_scores,
        temporal_positive,
        temporal_absolute,
        temporal_influence,
        temporal_expected,
        temporal_covered,
        prototype_scores,
    ) in zip(
        score_buffers,
        positive_score_buffers,
        absolute_score_buffers,
        influence_score_buffers,
        expected_buffers,
        causal_next_score_buffers,
        causal_next_positive_buffers,
        causal_next_absolute_buffers,
        causal_next_influence_buffers,
        causal_next_expected_buffers,
        causal_next_covered_buffers,
        causal_prototype_score_buffers,
    ):
        # Persistent pairs use the causal one-step-ahead residual.  Newly
        # appearing pairs fall back to masked current-relation surprise.
        final_scores = torch.where(
            temporal_covered,
            temporal_scores,
            current_scores,
        )
        final_positive = torch.where(
            temporal_covered,
            temporal_positive,
            current_positive,
        )
        final_absolute = torch.where(
            temporal_covered,
            temporal_absolute,
            current_absolute,
        )
        final_influence = torch.where(
            temporal_covered,
            temporal_influence,
            current_influence,
        )
        final_expected = torch.where(
            temporal_covered.unsqueeze(-1),
            temporal_expected,
            current_expected,
        )
        one_class_scores = (
            final_scores
            + RELATION_PROTOTYPE_SCORE_WEIGHT * prototype_scores
        )

        outputs.append(
            {
                "edge_scores": final_scores,
                "positive_edge_scores": final_positive,
                "absolute_edge_scores": final_absolute,
                "predictive_influence_scores": (
                    final_influence
                ),
                "expected_edge_attr": final_expected,
                "masked_current_edge_scores": (
                    current_scores
                ),
                "causal_next_edge_scores": (
                    temporal_scores
                ),
                "causal_next_coverage": temporal_covered,
                "causal_prototype_scores": prototype_scores,
                "one_class_edge_scores": one_class_scores,
            }
        )

    return outputs


def matched_next_edge_targets(
    current_graph: Any,
    next_graph: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = current_graph.edge_index.device
    edge_attr_dim = int(current_graph.edge_attr.size(1))

    next_keys = directed_vehicle_edge_keys(
        next_graph
    )

    next_targets = {
        key: next_graph.edge_attr[index]
        for index, key in enumerate(next_keys)
    }

    current_indices: list[int] = []
    targets: list[torch.Tensor] = []

    for index, key in enumerate(
        directed_vehicle_edge_keys(current_graph)
    ):
        target = next_targets.get(key)

        if target is None:
            continue

        current_indices.append(index)
        targets.append(target)

    if not current_indices:
        return (
            torch.empty(
                0,
                dtype=torch.long,
                device=device,
            ),
            torch.empty(
                (0, edge_attr_dim),
                dtype=current_graph.edge_attr.dtype,
                device=device,
            ),
        )

    return (
        torch.tensor(
            current_indices,
            dtype=torch.long,
            device=device,
        ),
        torch.stack(targets).to(device),
    )


def matched_persistent_edge_indices(
    current_graph: Any,
    next_graph: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = current_graph.edge_index.device

    next_index_by_key = {
        key: index
        for index, key in enumerate(
            directed_vehicle_edge_keys(next_graph)
        )
    }

    current_indices: list[int] = []
    next_indices: list[int] = []

    for index, key in enumerate(
        directed_vehicle_edge_keys(current_graph)
    ):
        next_index = next_index_by_key.get(key)

        if next_index is None:
            continue

        current_indices.append(index)
        next_indices.append(next_index)

    return (
        torch.tensor(
            current_indices,
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            next_indices,
            dtype=torch.long,
            device=next_graph.edge_index.device,
        ),
    )


def weighted_relation_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    if prediction.numel() == 0:
        return prediction.sum()

    loss = F.smooth_l1_loss(
        prediction,
        target,
        beta=0.05,
        reduction="none",
    )

    return (
        loss
        * channel_weights.view(1, -1)
    ).mean()


def relation_topology_auxiliary_loss(
    model: TemporalImmuneLocalizer,
    graph: Any,
    z: torch.Tensor,
    negative_generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    negative_edge_index = sample_negative_edges(
        edge_index=graph.edge_index,
        ptr=single_graph_ptr(graph),
        negative_ratio=NEGATIVE_RATIO,
        generator=negative_generator,
    )

    positive_logits = model.decode_topology(
        z,
        graph.edge_index,
    )

    positive_loss = F.binary_cross_entropy_with_logits(
        positive_logits,
        torch.ones_like(positive_logits),
    )

    positive_probability = torch.sigmoid(
        positive_logits
    ).mean()

    if negative_edge_index.numel() == 0:
        negative_loss = torch.zeros(
            (),
            device=z.device,
        )

        negative_probability = torch.zeros(
            (),
            device=z.device,
        )
    else:
        negative_logits = model.decode_topology(
            z,
            negative_edge_index,
        )

        negative_loss = F.binary_cross_entropy_with_logits(
            negative_logits,
            torch.zeros_like(negative_logits),
        )

        negative_probability = torch.sigmoid(
            negative_logits
        ).mean()

    return (
        positive_loss + negative_loss,
        positive_probability,
        negative_probability,
    )


def compute_relation_localizer_sequence_loss(
    model: TemporalImmuneLocalizer,
    sequence: list[Any],
    device: torch.device,
    mask_generator: torch.Generator,
    negative_generator: torch.Generator,
    mask_fraction: float,
    full_mask_probability: float,
    objective_mode: str = RELATION_OBJECTIVE_HYBRID,
    update_prototype: bool = False,
) -> RelationLossOutput:
    if objective_mode not in RELATION_OBJECTIVE_MODES:
        raise ValueError(
            f"Unknown relation objective mode: {objective_mode}"
        )
    device_sequence = [
        graph.clone().to(device)
        for graph in sequence
    ]

    edge_masks = [
        random_edge_mask(
            int(graph.edge_index.size(1)),
            mask_generator,
            device,
            mask_fraction,
            full_mask_probability,
        )
        for graph in device_sequence
    ]

    (
        temporal_embeddings,
        prior_edge_embeddings,
        updated_edge_embeddings,
    ) = model.encode_relation_sequence(
        device_sequence,
        device,
        edge_masks=edge_masks,
        return_updated_states=True,
    )

    channel_weights = (
        model.relation_channel_weights
        .to(device)
    )

    masked_losses: list[torch.Tensor] = []
    memoryless_masked_losses: list[
        torch.Tensor
    ] = []
    topology_losses: list[torch.Tensor] = []
    latent_losses: list[torch.Tensor] = []
    positive_probabilities: list[torch.Tensor] = []
    negative_probabilities: list[torch.Tensor] = []
    masked_edge_count = 0

    for (
        graph,
        z,
        prior_edge_state,
        updated_edge_state,
        edge_mask,
    ) in zip(
        device_sequence,
        temporal_embeddings,
        prior_edge_embeddings,
        updated_edge_embeddings,
        edge_masks,
    ):
        latent_component = z.pow(2).mean()
        if updated_edge_state.numel() > 0:
            latent_component = (
                latent_component
                + updated_edge_state.pow(2).mean()
            )
        latent_losses.append(latent_component)

        # A real ego sensor view may contain only ego for a frame. US-101's
        # omniscient dense tracks rarely exercised this case, but reducing an
        # empty edge tensor would produce NaN and poison the whole sequence.
        if graph.edge_attr.numel() == 0:
            continue

        predicted_relation = model.predict_relation_attr(
            z,
            graph.edge_index,
            prior_edge_state,
        )
        memoryless_relation = (
            model.predict_memoryless_relation_attr(
                z,
                graph.edge_index,
            )
        )

        masked_loss = weighted_relation_smooth_l1(
            predicted_relation[edge_mask],
            graph.edge_attr[edge_mask],
            channel_weights,
        )

        topology_loss, positive_probability, negative_probability = (
            relation_topology_auxiliary_loss(
                model=model,
                graph=graph,
                z=z,
                negative_generator=negative_generator,
            )
        )

        masked_losses.append(masked_loss)
        memoryless_masked_losses.append(
            weighted_relation_smooth_l1(
                memoryless_relation[edge_mask],
                graph.edge_attr[edge_mask],
                channel_weights,
            )
        )
        topology_losses.append(topology_loss)
        positive_probabilities.append(positive_probability)
        negative_probabilities.append(negative_probability)
        masked_edge_count += int(edge_mask.sum().item())

    next_losses: list[torch.Tensor] = []
    memoryless_next_losses: list[
        torch.Tensor
    ] = []
    compactness_losses: list[torch.Tensor] = []
    transition_states: list[torch.Tensor] = []
    next_edge_count = 0
    persistent_pair_count = 0

    for frame_index in range(
        len(device_sequence) - 1
    ):
        current_graph = device_sequence[frame_index]
        next_graph = device_sequence[frame_index + 1]
        current_z = temporal_embeddings[frame_index]
        current_edge_state = updated_edge_embeddings[
            frame_index
        ]
        next_edge_state = updated_edge_embeddings[
            frame_index + 1
        ]

        current_indices, next_targets = (
            matched_next_edge_targets(
                current_graph,
                next_graph,
            )
        )

        predicted_next = model.predict_next_relation_attr(
            current_z,
            current_graph.edge_index,
            current_edge_state,
        )
        memoryless_next = (
            model.predict_memoryless_next_relation_attr(
                current_z,
                current_graph.edge_index,
            )
        )

        if current_indices.numel() > 0:
            next_loss = weighted_relation_smooth_l1(
                predicted_next[current_indices],
                next_targets,
                channel_weights,
            )
            next_losses.append(next_loss)
            memoryless_next_losses.append(
                weighted_relation_smooth_l1(
                    memoryless_next[current_indices],
                    next_targets,
                    channel_weights,
                )
            )
            next_edge_count += int(current_indices.numel())

        current_pair_indices, next_pair_indices = (
            matched_persistent_edge_indices(
                current_graph,
                next_graph,
            )
        )

        if current_pair_indices.numel() == 0:
            compactness_loss = current_z.sum() * 0.0
        else:
            transition_state = (
                next_edge_state[next_pair_indices]
                - current_edge_state[current_pair_indices]
            )
            transition_states.append(transition_state)
            compactness_loss = F.smooth_l1_loss(
                current_edge_state[
                    current_pair_indices
                ],
                next_edge_state[
                    next_pair_indices
                ],
                beta=0.10,
            )

        compactness_losses.append(compactness_loss)
        persistent_pair_count += int(current_pair_indices.numel())

    zero_loss = temporal_embeddings[0].sum() * 0.0
    masked_edge_loss = (
        torch.stack(masked_losses).mean()
        if masked_losses
        else zero_loss
    )
    memoryless_masked_loss = (
        torch.stack(memoryless_masked_losses).mean()
        if memoryless_masked_losses
        else zero_loss
    )

    topology_loss = (
        torch.stack(topology_losses).mean()
        if topology_losses
        else zero_loss
    )

    latent_regularization = torch.stack(
        latent_losses
    ).mean()

    positive_probability = (
        torch.stack(positive_probabilities).mean()
        if positive_probabilities
        else zero_loss
    )

    negative_probability = (
        torch.stack(negative_probabilities).mean()
        if negative_probabilities
        else zero_loss
    )

    if next_losses:
        next_edge_loss = torch.stack(
            next_losses
        ).mean()
        memoryless_next_loss = torch.stack(
            memoryless_next_losses
        ).mean()
    else:
        next_edge_loss = masked_edge_loss * 0.0
        memoryless_next_loss = (
            masked_edge_loss * 0.0
        )

    if objective_mode in {
        RELATION_OBJECTIVE_CAUSAL_FUTURE,
        RELATION_OBJECTIVE_ONE_CLASS_PROTOTYPE,
    }:
        # The current edge target is algebraically recoverable from current
        # node position/velocity features.  Keep computing it for diagnostics,
        # but do not optimize that shortcut in the causal immune objective.
        memoryless_reference_loss = memoryless_next_loss
    else:
        memoryless_reference_loss = (
            memoryless_masked_loss
            + memoryless_next_loss
        )

    if compactness_losses:
        compactness_loss = torch.stack(
            compactness_losses
        ).mean()
    else:
        compactness_loss = masked_edge_loss * 0.0

    if transition_states:
        all_transition_states = torch.cat(transition_states, dim=0)
        if (
            objective_mode == RELATION_OBJECTIVE_ONE_CLASS_PROTOTYPE
            and update_prototype
        ):
            model.update_safe_relation_prototype(all_transition_states)
        prototype_distances = model.safe_relation_prototype_distance(
            all_transition_states
        )
        prototype_compactness_loss = (
            prototype_distances.pow(2).mean()
            if prototype_distances.numel() > 0
            else masked_edge_loss * 0.0
        )
    else:
        prototype_compactness_loss = masked_edge_loss * 0.0

    total_loss = (
        RELATION_NEXT_EDGE_WEIGHT
        * next_edge_loss
        + RELATION_MEMORYLESS_AUX_WEIGHT
        * memoryless_reference_loss
        + RELATION_COMPACTNESS_WEIGHT
        * compactness_loss
        + RELATION_TOPOLOGY_WEIGHT
        * topology_loss
        + RELATION_LATENT_REGULARIZATION_WEIGHT
        * latent_regularization
    )

    if objective_mode == RELATION_OBJECTIVE_ONE_CLASS_PROTOTYPE:
        total_loss = (
            total_loss
            + RELATION_PROTOTYPE_COMPACTNESS_WEIGHT
            * prototype_compactness_loss
        )

    if objective_mode == RELATION_OBJECTIVE_HYBRID:
        total_loss = (
            total_loss
            + RELATION_MASKED_EDGE_WEIGHT
            * masked_edge_loss
        )

    return RelationLossOutput(
        total=total_loss,
        masked_edge=masked_edge_loss,
        next_edge=next_edge_loss,
        memoryless_reference=(
            memoryless_reference_loss
        ),
        compactness=compactness_loss,
        prototype_compactness=prototype_compactness_loss,
        topology=topology_loss,
        latent_reg=latent_regularization,
        positive_probability=positive_probability,
        negative_probability=negative_probability,
        masked_edge_count=float(masked_edge_count),
        next_edge_count=float(next_edge_count),
        persistent_pair_count=float(persistent_pair_count),
    )


def train_relation_localizer_one_epoch(
    model: TemporalImmuneLocalizer,
    sequences: list[list[Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    objective_mode: str = RELATION_OBJECTIVE_HYBRID,
) -> dict[str, float]:
    model.train()

    accumulator = RelationMetricAccumulator()
    shuffled_indices = list(range(len(sequences)))

    random.Random(
        RANDOM_SEED + epoch
    ).shuffle(shuffled_indices)

    mask_generator = torch.Generator(
        device="cpu"
    )

    mask_generator.manual_seed(
        RANDOM_SEED
        + epoch * 10_000
        + 201
    )

    negative_generator = torch.Generator(
        device="cpu"
    )

    negative_generator.manual_seed(
        RANDOM_SEED
        + epoch * 10_000
        + 202
    )

    mask_fraction = relation_mask_fraction(
        epoch,
        total_epochs,
    )

    for sequence_index in shuffled_indices:
        sequence = sequences[sequence_index]
        optimizer.zero_grad(
            set_to_none=True
        )

        loss_output = compute_relation_localizer_sequence_loss(
            model=model,
            sequence=sequence,
            device=device,
            mask_generator=mask_generator,
            negative_generator=negative_generator,
            mask_fraction=mask_fraction,
            full_mask_probability=(
                RELATION_FULL_MASK_PROBABILITY
            ),
            objective_mode=objective_mode,
            update_prototype=(
                objective_mode
                == RELATION_OBJECTIVE_ONE_CLASS_PROTOTYPE
            ),
        )

        if not torch.isfinite(loss_output.total):
            raise FloatingPointError(
                f"{display_model_name(model.config.model_name)} "
                "训练 loss 出现 NaN/Inf。"
            )

        loss_output.total.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_NORM,
        )

        optimizer.step()

        accumulator.update(
            loss_output,
            len(sequence),
        )

    metrics = accumulator.mean()
    metrics["mask_fraction"] = mask_fraction
    return metrics


@torch.no_grad()
def evaluate_relation_localizer_one_epoch(
    model: TemporalImmuneLocalizer,
    sequences: list[list[Any]],
    device: torch.device,
    epoch: int,
    total_epochs: int,
    objective_mode: str = RELATION_OBJECTIVE_HYBRID,
) -> dict[str, float]:
    model.eval()

    accumulator = RelationMetricAccumulator()

    mask_generator = torch.Generator(
        device="cpu"
    )

    mask_generator.manual_seed(
        RANDOM_SEED
        + epoch * 10_000
        + 203
    )

    negative_generator = torch.Generator(
        device="cpu"
    )

    negative_generator.manual_seed(
        RANDOM_SEED
        + epoch * 10_000
        + 204
    )

    del total_epochs

    # Keep validation difficulty fixed so checkpoint selection is comparable
    # across epochs and directly measures the full-mask deployment condition.
    mask_fraction = RELATION_MASK_END_FRACTION

    for sequence in sequences:
        loss_output = compute_relation_localizer_sequence_loss(
            model=model,
            sequence=sequence,
            device=device,
            mask_generator=mask_generator,
            negative_generator=negative_generator,
            mask_fraction=mask_fraction,
            full_mask_probability=0.0,
            objective_mode=objective_mode,
        )

        accumulator.update(
            loss_output,
            len(sequence),
        )

    metrics = accumulator.mean()
    metrics["mask_fraction"] = mask_fraction
    return metrics


# ============================================================
# Checkpoint and history output
# ============================================================

def checkpoint_path(
    model_name: str,
) -> Path:
    if model_name == "Temporal_Immune_GAE":
        return (
            CHECKPOINT_DIR
            / f"Temporal_Immune_GAE_{DATASET_TAG}_temporal_seq.pth"
        )

    if model_name == RELATION_MODEL_NAME:
        return (
            CHECKPOINT_DIR
            / (
                "Temporal_Immune_Localizer_"
                f"{DATASET_TAG}_temporal_seq.pth"
            )
        )

    if model_name == RELATION_GCN_ABLATION_NAME:
        return (
            CHECKPOINT_DIR
            / (
                "Temporal_Immune_Localizer_without_Immune_Encoder"
                f"_{DATASET_TAG}_temporal_seq.pth"
            )
        )

    if model_name == RELATION_EDGE_MEMORY_ABLATION_NAME:
        return (
            CHECKPOINT_DIR
            / (
                "TRAIL_without_Edge_Memory_"
                f"{DATASET_TAG}_temporal_seq.pth"
            )
        )

    if model_name == LEGACY_RELATION_MODEL_NAME:
        return (
            CHECKPOINT_DIR
            / (
                "Temporal_Relation_Localizer_"
                f"{DATASET_TAG}_temporal_seq.pth"
            )
        )

    return (
        CHECKPOINT_DIR
        / f"{model_name}_{DATASET_TAG}_temporal.pth"
    )


def model_history_path(
    model_name: str,
) -> Path:
    return (
        RESULT_DIR
        / f"{model_name}_history.csv"
    )


def save_model_checkpoint(
    *,
    model: BaseGraphAutoencoder,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    epoch: int,
    best_val_loss: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    dataset_meta: dict[str, Any],
    node_normalization: dict[str, Any],
) -> None:
    path = checkpoint_path(
        config.model_name
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_name": config.model_name,
            "model_config": asdict(config),
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "best_epoch": epoch,
            "best_val_loss": (
                best_val_loss
            ),
            "train_metrics": (
                train_metrics
            ),
            "val_metrics": (
                val_metrics
            ),
            "training_configuration": {
                "epochs": (
                    RELATION_LOCALIZER_EPOCHS
                    if is_relation_localizer_name(
                        config.model_name
                    )
                    else
                    TEMPORAL_EPOCHS
                    if config.model_name == "Temporal_Immune_GAE"
                    else EPOCHS
                ),
                "batch_size": BATCH_SIZE,
                "learning_rate": (
                    LEARNING_RATE
                ),
                "weight_decay": (
                    WEIGHT_DECAY
                ),
                "negative_ratio": (
                    NEGATIVE_RATIO
                ),
                "topology_weight": (
                    TOPOLOGY_WEIGHT
                ),
                "edge_attribute_weight": (
                    EDGE_ATTRIBUTE_WEIGHT
                ),
                "node_reconstruction_weight": (
                    NODE_RECONSTRUCTION_WEIGHT
                ),
                "latent_regularization_weight": (
                    LATENT_REGULARIZATION_WEIGHT
                ),
                "relation_mask_start_fraction": (
                    RELATION_MASK_START_FRACTION
                    if is_relation_localizer_name(
                        config.model_name
                    )
                    else None
                ),
                "relation_mask_end_fraction": (
                    RELATION_MASK_END_FRACTION
                    if is_relation_localizer_name(
                        config.model_name
                    )
                    else None
                ),
                "relation_mask_curriculum_epochs": (
                    RELATION_MASK_CURRICULUM_EPOCHS
                    if is_relation_localizer_name(
                        config.model_name
                    )
                    else None
                ),
                "relation_full_mask_probability": (
                    RELATION_FULL_MASK_PROBABILITY
                    if is_relation_localizer_name(
                        config.model_name
                    )
                    else None
                ),
                "relation_surprise_mode": (
                    RELATION_SURPRISE_MODE
                    if is_relation_localizer_name(
                        config.model_name
                    )
                    else None
                ),
                "relation_inference_mask_groups": (
                    RELATION_INFERENCE_MASK_GROUPS
                    if is_relation_localizer_name(
                        config.model_name
                    )
                    else None
                ),
                "relation_min_epochs_before_stop": (
                    RELATION_MIN_EPOCHS_BEFORE_STOP
                    if is_relation_localizer_name(
                        config.model_name
                    )
                    else None
                ),
                "relation_early_stopping_patience": (
                    RELATION_EARLY_STOPPING_PATIENCE
                    if is_relation_localizer_name(
                        config.model_name
                    )
                    else None
                ),
                "relation_masked_edge_weight": (
                    RELATION_MASKED_EDGE_WEIGHT
                    if is_relation_localizer_name(
                        config.model_name
                    )
                    else None
                ),
                "relation_next_edge_weight": (
                    RELATION_NEXT_EDGE_WEIGHT
                    if is_relation_localizer_name(
                        config.model_name
                    )
                    else None
                ),
                "relation_memoryless_aux_weight": (
                    RELATION_MEMORYLESS_AUX_WEIGHT
                    if is_relation_localizer_name(
                        config.model_name
                    )
                    else None
                ),
                "relation_compactness_weight": (
                    RELATION_COMPACTNESS_WEIGHT
                    if is_relation_localizer_name(
                        config.model_name
                    )
                    else None
                ),
                "relation_topology_weight": (
                    RELATION_TOPOLOGY_WEIGHT
                    if is_relation_localizer_name(
                        config.model_name
                    )
                    else None
                ),
                "positive_topology_edge_weighting": "uniform",
                "random_seed": (
                    RANDOM_SEED
                ),
                "temporal_sequence_length": (
                    TEMPORAL_SEQUENCE_LENGTH
                    if config.model_name in {
                        "Temporal_Immune_GAE",
                        RELATION_MODEL_NAME,
                        RELATION_GCN_ABLATION_NAME,
                        RELATION_EDGE_MEMORY_ABLATION_NAME,
                        LEGACY_RELATION_MODEL_NAME,
                    }
                    else None
                ),
                "temporal_sequence_stride": (
                    TEMPORAL_SEQUENCE_STRIDE
                    if config.model_name in {
                        "Temporal_Immune_GAE",
                        RELATION_MODEL_NAME,
                        RELATION_GCN_ABLATION_NAME,
                        RELATION_EDGE_MEMORY_ABLATION_NAME,
                        LEGACY_RELATION_MODEL_NAME,
                    }
                    else None
                ),
                "temporal_train_on_full_clips": (
                    TEMPORAL_TRAIN_ON_FULL_CLIPS
                    if config.model_name in {
                        "Temporal_Immune_GAE",
                        RELATION_MODEL_NAME,
                        RELATION_GCN_ABLATION_NAME,
                        RELATION_EDGE_MEMORY_ABLATION_NAME,
                        LEGACY_RELATION_MODEL_NAME,
                    }
                    else None
                ),
            },
            "dataset_meta": dataset_meta,
            "node_normalization": {
                "feature_names": (
                    node_normalization[
                        "feature_names"
                    ]
                ),
                "mean": (
                    node_normalization[
                        "mean"
                    ].detach().cpu()
                ),
                "std": (
                    node_normalization[
                        "std"
                    ].detach().cpu()
                ),
                "fit_split": (
                    node_normalization[
                        "fit_split"
                    ]
                ),
            },
        },
        path,
    )


def save_history_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Different model families log different metrics.  Build one stable union
    # of all keys instead of assuming that the first row contains every field.
    fieldnames = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row.keys()
        )
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Single-model training
# ============================================================

def train_model(
    *,
    config: ModelConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    dataset_meta: dict[str, Any],
    node_normalization: dict[str, Any],
) -> dict[str, Any]:
    set_global_seed(
        RANDOM_SEED
    )

    model = build_model(
        config
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=7,
            min_lr=1e-5,
        )
    )

    parameter_count = count_parameters(
        model
    )

    print("\n" + "=" * 72)
    print(
        f"🚀 开始训练模型："
        f"{config.model_name}"
    )
    print(
        f"可训练参数量："
        f"{parameter_count:,}"
    )
    print("=" * 72)

    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    history_rows: list[
        dict[str, Any]
    ] = []

    training_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
        )

        scheduler.step(
            val_metrics["total"]
        )

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        history_row = {
            "model_name": (
                config.model_name
            ),
            "epoch": epoch,
            "learning_rate": (
                current_lr
            ),
            "epoch_seconds": (
                epoch_seconds
            ),
        }

        for split_name, metrics in [
            ("train", train_metrics),
            ("val", val_metrics),
        ]:
            for metric_name, value in (
                metrics.items()
            ):
                history_row[
                    f"{split_name}_{metric_name}"
                ] = value

        history_rows.append(
            history_row
        )

        improved = (
            best_val_loss
            - val_metrics["total"]
            > MIN_IMPROVEMENT
        )

        if improved:
            best_val_loss = (
                val_metrics["total"]
            )
            best_epoch = epoch
            epochs_without_improvement = 0

            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                best_val_loss=best_val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                dataset_meta=dataset_meta,
                node_normalization=(
                    node_normalization
                ),
            )
        else:
            epochs_without_improvement += 1

        should_print = (
            epoch == 1
            or epoch % PRINT_EVERY == 0
            or epoch == EPOCHS
            or improved
        )

        if should_print:
            improvement_marker = (
                " ★ best"
                if improved
                else ""
            )

            print(
                f"[{config.model_name}] "
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Train={train_metrics['total']:.5f} | "
                f"Val={val_metrics['total']:.5f} | "
                f"ValTopo={val_metrics['topology']:.5f} | "
                f"ValAttr={val_metrics['edge_attr']:.5f} | "
                f"ValNode={val_metrics['node']:.5f} | "
                f"PosP={val_metrics['positive_probability']:.4f} | "
                f"NegP={val_metrics['negative_probability']:.4f} | "
                f"LR={current_lr:.2e} | "
                f"{epoch_seconds:.1f}s"
                f"{improvement_marker}"
            )

        if (
            epoch >= MIN_EPOCHS_BEFORE_STOP
            and epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"⏹️ Early stopping："
                f"{EARLY_STOPPING_PATIENCE} "
                f"个 epoch 未改善。"
            )
            break

    total_seconds = (
        time.perf_counter()
        - training_start
    )

    save_history_csv(
        history_rows,
        model_history_path(
            config.model_name
        ),
    )

    checkpoint = torch.load(
        checkpoint_path(
            config.model_name
        ),
        map_location="cpu",
        weights_only=False,
    )

    summary = {
        "model_name": (
            config.model_name
        ),
        "parameter_count": (
            parameter_count
        ),
        "best_epoch": int(
            checkpoint["best_epoch"]
        ),
        "best_val_loss": float(
            checkpoint["best_val_loss"]
        ),
        "best_train_metrics": (
            checkpoint["train_metrics"]
        ),
        "best_val_metrics": (
            checkpoint["val_metrics"]
        ),
        "epochs_completed": len(
            history_rows
        ),
        "training_seconds": (
            total_seconds
        ),
        "checkpoint_path": str(
            checkpoint_path(
                config.model_name
            )
        ),
        "history_path": str(
            model_history_path(
                config.model_name
            )
        ),
    }

    print(
        f"💾 最佳权重："
        f"{summary['checkpoint_path']}"
    )

    print(
        f"✅ {config.model_name} 完成："
        f"best epoch={summary['best_epoch']}, "
        f"best val={summary['best_val_loss']:.6f}, "
        f"time={total_seconds / 60:.1f} min"
    )

    return {
        "summary": summary,
        "history": history_rows,
    }


def train_temporal_model(
    *,
    config: ModelConfig,
    train_sequences: list[list[Any]],
    val_sequences: list[list[Any]],
    device: torch.device,
    dataset_meta: dict[str, Any],
    node_normalization: dict[str, Any],
    sequence_summary: dict[str, Any],
) -> dict[str, Any]:
    set_global_seed(
        RANDOM_SEED
    )

    model = TemporalImmuneGAE(
        config
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=7,
        min_lr=1e-5,
    )

    parameter_count = count_parameters(
        model
    )

    print("\n" + "=" * 72)
    print("🚀 开始训练模型：Temporal_Immune_GAE")
    print(f"可训练参数量：{parameter_count:,}")
    print(
        f"Temporal train sequences: {len(train_sequences)} | "
        f"val sequences: {len(val_sequences)} | "
        f"minimum length: {TEMPORAL_SEQUENCE_LENGTH}"
    )
    print("=" * 72)

    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []

    training_start = time.perf_counter()

    for epoch in range(1, TEMPORAL_EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_temporal_one_epoch(
            model=model,
            sequences=train_sequences,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        val_metrics = evaluate_temporal_one_epoch(
            model=model,
            sequences=val_sequences,
            device=device,
            epoch=epoch,
        )

        scheduler.step(
            val_metrics["total"]
        )

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        epoch_seconds = time.perf_counter() - epoch_start

        history_row = {
            "model_name": config.model_name,
            "epoch": epoch,
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
            "train_sequence_count": len(train_sequences),
            "val_sequence_count": len(val_sequences),
        }

        for split_name, metrics in [
            ("train", train_metrics),
            ("val", val_metrics),
        ]:
            for metric_name, value in metrics.items():
                history_row[f"{split_name}_{metric_name}"] = value

        history_rows.append(history_row)

        improved = best_val_loss - val_metrics["total"] > MIN_IMPROVEMENT

        if improved:
            best_val_loss = val_metrics["total"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                best_val_loss=best_val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                dataset_meta={
                    **dataset_meta,
                    "temporal_sequence_training": sequence_summary,
                },
                node_normalization=node_normalization,
            )
        else:
            epochs_without_improvement += 1

        should_print = (
            epoch == 1
            or epoch % PRINT_EVERY == 0
            or epoch == TEMPORAL_EPOCHS
            or improved
        )

        if should_print:
            improvement_marker = " ★ best" if improved else ""

            print(
                f"[{config.model_name}] "
                f"Epoch {epoch:03d}/{TEMPORAL_EPOCHS} | "
                f"Train={train_metrics['total']:.5f} | "
                f"Val={val_metrics['total']:.5f} | "
                f"ValTopo={val_metrics['topology']:.5f} | "
                f"ValAttr={val_metrics['edge_attr']:.5f} | "
                f"ValNode={val_metrics['node']:.5f} | "
                f"PosP={val_metrics['positive_probability']:.4f} | "
                f"NegP={val_metrics['negative_probability']:.4f} | "
                f"LR={current_lr:.2e} | "
                f"{epoch_seconds:.1f}s"
                f"{improvement_marker}"
            )

        if (
            epoch >= MIN_EPOCHS_BEFORE_STOP
            and epochs_without_improvement >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"⏹️ Temporal early stopping："
                f"{EARLY_STOPPING_PATIENCE} 个 epoch 未改善。"
            )
            break

    total_seconds = time.perf_counter() - training_start

    save_history_csv(
        history_rows,
        model_history_path(config.model_name),
    )

    checkpoint = torch.load(
        checkpoint_path(config.model_name),
        map_location="cpu",
        weights_only=False,
    )

    summary = {
        "model_name": config.model_name,
        "parameter_count": parameter_count,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "best_train_metrics": checkpoint["train_metrics"],
        "best_val_metrics": checkpoint["val_metrics"],
        "epochs_completed": len(history_rows),
        "training_seconds": total_seconds,
        "checkpoint_path": str(checkpoint_path(config.model_name)),
        "history_path": str(model_history_path(config.model_name)),
        "temporal_sequence_training": sequence_summary,
    }

    print(f"💾 Temporal 最佳权重：{summary['checkpoint_path']}")
    print(
        f"✅ {config.model_name} 完成："
        f"best epoch={summary['best_epoch']}, "
        f"best val={summary['best_val_loss']:.6f}, "
        f"time={total_seconds / 60:.1f} min"
    )

    return {
        "summary": summary,
        "history": history_rows,
    }


def train_relation_localizer_model(
    *,
    config: ModelConfig,
    train_sequences: list[list[Any]],
    val_sequences: list[list[Any]],
    device: torch.device,
    dataset_meta: dict[str, Any],
    node_normalization: dict[str, Any],
    sequence_summary: dict[str, Any],
    epochs: int = RELATION_LOCALIZER_EPOCHS,
) -> dict[str, Any]:
    set_global_seed(
        RANDOM_SEED
    )

    model = build_model(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=7,
        min_lr=1e-5,
    )

    parameter_count = count_parameters(
        model
    )

    print("\n" + "=" * 72)
    print(
        f"🚀 开始训练模型："
        f"{display_model_name(config.model_name)}"
    )
    print(f"可训练参数量：{parameter_count:,}")
    print(
        f"Relation train sequences: {len(train_sequences)} | "
        f"val sequences: {len(val_sequences)} | "
        f"epochs: {epochs}"
    )
    print(
        "目标：masked edge prediction + next-step relation prediction"
    )
    print("=" * 72)

    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []

    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_relation_localizer_one_epoch(
            model=model,
            sequences=train_sequences,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )

        val_metrics = evaluate_relation_localizer_one_epoch(
            model=model,
            sequences=val_sequences,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )

        scheduler.step(
            val_metrics["total"]
        )

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        epoch_seconds = time.perf_counter() - epoch_start

        history_row = {
            "model_name": config.model_name,
            "epoch": epoch,
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
            "train_sequence_count": len(train_sequences),
            "val_sequence_count": len(val_sequences),
        }

        for split_name, metrics in [
            ("train", train_metrics),
            ("val", val_metrics),
        ]:
            for metric_name, value in metrics.items():
                history_row[f"{split_name}_{metric_name}"] = value

        history_rows.append(history_row)

        improved = (
            best_val_loss
            - val_metrics["total"]
            > MIN_IMPROVEMENT
        )

        if improved:
            best_val_loss = val_metrics["total"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                best_val_loss=best_val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                dataset_meta={
                    **dataset_meta,
                    "temporal_sequence_training": sequence_summary,
                    "phase3_objective": (
                        "masked_edge_prediction_and_next_step_relation_prediction"
                    ),
                },
                node_normalization=node_normalization,
            )
        else:
            epochs_without_improvement += 1

        should_print = (
            epoch == 1
            or epoch % PRINT_EVERY == 0
            or epoch == epochs
            or improved
        )

        if should_print:
            improvement_marker = " ★ best" if improved else ""

            print(
                f"[{config.model_name}] "
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train={train_metrics['total']:.5f} | "
                f"Val={val_metrics['total']:.5f} | "
                f"ValMasked={val_metrics['masked_edge']:.5f} | "
                f"ValNext={val_metrics['next_edge']:.5f} | "
                f"ValRef={val_metrics['memoryless_reference']:.5f} | "
                f"ValCompact={val_metrics['compactness']:.5f} | "
                f"ValTopo={val_metrics['topology']:.5f} | "
                f"Mask={val_metrics['mask_fraction']:.2f} | "
                f"PosP={val_metrics['positive_probability']:.4f} | "
                f"NegP={val_metrics['negative_probability']:.4f} | "
                f"LR={current_lr:.2e} | "
                f"{epoch_seconds:.1f}s"
                f"{improvement_marker}"
            )

        if (
            epoch >= RELATION_MIN_EPOCHS_BEFORE_STOP
            and epochs_without_improvement
            >= RELATION_EARLY_STOPPING_PATIENCE
        ):
            print(
                f"⏹️ {display_model_name(config.model_name)} "
                "early stopping："
                f"{RELATION_EARLY_STOPPING_PATIENCE} 个 epoch 未改善。"
            )
            break

    total_seconds = time.perf_counter() - training_start

    save_history_csv(
        history_rows,
        model_history_path(config.model_name),
    )

    checkpoint = torch.load(
        checkpoint_path(config.model_name),
        map_location="cpu",
        weights_only=False,
    )

    summary = {
        "model_name": config.model_name,
        "parameter_count": parameter_count,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "best_train_metrics": checkpoint["train_metrics"],
        "best_val_metrics": checkpoint["val_metrics"],
        "epochs_completed": len(history_rows),
        "training_seconds": total_seconds,
        "checkpoint_path": str(checkpoint_path(config.model_name)),
        "history_path": str(model_history_path(config.model_name)),
        "temporal_sequence_training": sequence_summary,
        "objective": {
            "masked_edge_prediction": True,
            "next_step_relation_prediction": True,
            "directed_edge_temporal_memory": True,
            "causal_pair_memory": (
                "current relation uses prior pair state; "
                "next relation uses updated pair state"
            ),
            "persistent_edge_scoring": (
                "causal one-step-ahead absolute surprise"
            ),
            "new_edge_scoring_fallback": (
                "masked current-relation absolute surprise"
            ),
            "memoryless_predictive_reference": True,
            "learned_predictive_influence": (
                "memoryless absolute error minus "
                "edge-memory absolute error"
            ),
            "handwritten_interaction_strength": False,
            "relation_compactness": True,
            "topology_auxiliary": True,
            "mask_curriculum": {
                "start_fraction": RELATION_MASK_START_FRACTION,
                "end_fraction": RELATION_MASK_END_FRACTION,
                "curriculum_epochs": (
                    RELATION_MASK_CURRICULUM_EPOCHS
                ),
                "full_mask_probability": (
                    RELATION_FULL_MASK_PROBABILITY
                ),
            },
            "early_stopping": {
                "minimum_epochs": (
                    RELATION_MIN_EPOCHS_BEFORE_STOP
                ),
                "patience": (
                    RELATION_EARLY_STOPPING_PATIENCE
                ),
            },
            "surprise_mode": RELATION_SURPRISE_MODE,
            "positive_surprise_weight": (
                RELATION_POSITIVE_SURPRISE_WEIGHT
            ),
            "absolute_surprise_weight": (
                RELATION_ABSOLUTE_SURPRISE_WEIGHT
            ),
            "uses_immune_encoder": (
                config.model_name
                != RELATION_GCN_ABLATION_NAME
            ),
            "uses_directed_edge_memory": (
                config.model_name
                != RELATION_EDGE_MEMORY_ABLATION_NAME
            ),
            "uses_ood_training_labels": False,
        },
    }

    print(f"💾 Relation Localizer 最佳权重：{summary['checkpoint_path']}")
    print(
        f"✅ {config.model_name} 完成："
        f"best epoch={summary['best_epoch']}, "
        f"best val={summary['best_val_loss']:.6f}, "
        f"time={total_seconds / 60:.1f} min"
    )

    return {
        "summary": summary,
        "history": history_rows,
    }


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train TRAIL on normal temporal relation sequences."
        )
    )

    parser.add_argument(
        "--dataset-profile",
        choices=sorted(DATASET_PROFILES),
        default="us101",
        help=(
            "Select the dataset path, relation schema, and isolated output "
            "directories. Counterfactual/OOD graphs are never used for "
            "optimization."
        ),
    )

    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Optional dataset override. The selected profile still supplies "
            "the expected schema unless --expected-relation-schema is set."
        ),
    )

    parser.add_argument(
        "--expected-relation-schema",
        type=str,
        default=None,
        help="Optional explicit relation-schema override.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional checkpoint directory override.",
    )

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Optional training-result directory override.",
    )

    parser.add_argument(
        "--only",
        choices=[
            "all",
            "relation-localizer",
            "relation-gcn-ablation",
            "relation-edge-memory-ablation",
            "relation-suite",
        ],
        default="all",
        help=(
            "Train TRAIL or one of its explicit ablations."
        ),
    )

    parser.add_argument(
        "--relation-epochs",
        type=int,
        default=RELATION_LOCALIZER_EPOCHS,
        help=(
            "Epoch count for relation-localizer models."
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Load the selected dataset, verify its schema and normal-only "
            "temporal splits, then exit without training or writing weights."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_dataset_profile(args)

    set_global_seed(
        RANDOM_SEED
    )

    device = choose_device(
        DEVICE_MODE
    )

    print(
        f"--- 1. 加载 {DATASET_LABEL} 图数据集 ---"
    )

    package = load_dataset()

    train_normal = package[
        "train_normal"
    ]

    val_normal = package[
        "val_normal"
    ]

    edge_normalization = package[
        "edge_normalization"
    ]

    dataset_meta = {
        **package["meta"],
        "dataset_profile": args.dataset_profile,
        "dataset_path": str(DATASET_PATH),
        "edge_normalization": (
            edge_normalization
        ),
    }

    node_normalization = package[
        "node_normalization"
    ]

    in_channels = int(
        train_normal[0].x.size(1)
    )

    edge_attr_dim = int(
        train_normal[0].edge_attr.size(1)
    )

    print(
        f"✅ train-normal graphs: "
        f"{len(train_normal)}"
    )

    print(
        f"✅ val-normal graphs: "
        f"{len(val_normal)}"
    )

    print(
        f"节点特征维度 in_channels = "
        f"{in_channels}"
    )

    print(
        f"边属性维度 edge_attr_dim = "
        f"{edge_attr_dim}"
    )

    print(
        f"使用设备：{device}"
    )

    print(
        f"relation schema："
        f"{EXPECTED_RELATION_SCHEMA_VERSION}"
    )

    print(
        "训练语义：仅使用 normal_graphs；"
        "paired counterfactual/OOD graphs 只保留给评估。"
    )

    if args.validate_only:
        (
            temporal_train_sequences,
            temporal_val_sequences,
            temporal_sequence_summary,
        ) = build_temporal_training_sequences(package)

        non_normal_training_graphs = [
            graph
            for sequence in (
                temporal_train_sequences
                + temporal_val_sequences
            )
            for graph in sequence
            if str(
                getattr(graph, "anomaly_type", "Normal")
            ) != "Normal"
        ]
        if non_normal_training_graphs:
            raise RuntimeError(
                "normal-only audit failed: temporal training sequences "
                "contain anomaly graphs"
            )

        print("\n✅ Dataset validation complete")
        print(
            f"normal-only train sequences: "
            f"{len(temporal_train_sequences)}"
        )
        print(
            f"normal-only validation sequences: "
            f"{len(temporal_val_sequences)}"
        )
        print(
            "counterfactual/OOD graphs used for optimization: 0"
        )
        print(
            f"sequence summary: {temporal_sequence_summary}"
        )
        return

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.only in {
        "relation-localizer",
        "relation-gcn-ablation",
        "relation-edge-memory-ablation",
        "relation-suite",
    }:
        print(
            "\n--- 2. 构建 normal sequence windows 并训练 relation localizer suite ---"
        )

        (
            temporal_train_sequences,
            temporal_val_sequences,
            temporal_sequence_summary,
        ) = build_temporal_training_sequences(package)

        print(
            f"✅ relation train sequences: "
            f"{len(temporal_train_sequences)}"
        )

        print(
            f"✅ relation val sequences: "
            f"{len(temporal_val_sequences)}"
        )

        relation_model_names = {
            "relation-localizer": [
                RELATION_MODEL_NAME
            ],
            "relation-gcn-ablation": [
                RELATION_GCN_ABLATION_NAME
            ],
            "relation-edge-memory-ablation": [
                RELATION_EDGE_MEMORY_ABLATION_NAME
            ],
            "relation-suite": [
                RELATION_GCN_ABLATION_NAME,
                RELATION_EDGE_MEMORY_ABLATION_NAME,
                RELATION_MODEL_NAME,
            ],
        }[args.only]

        relation_results = []

        for relation_model_name in relation_model_names:
            relation_config = ModelConfig(
                model_name=relation_model_name,
                in_channels=in_channels,
                edge_attr_dim=edge_attr_dim,
            )

            relation_results.append(
                train_relation_localizer_model(
                    config=relation_config,
                    train_sequences=temporal_train_sequences,
                    val_sequences=temporal_val_sequences,
                    device=device,
                    dataset_meta=dataset_meta,
                    node_normalization=node_normalization,
                    sequence_summary=temporal_sequence_summary,
                    epochs=args.relation_epochs,
                )
            )

        relation_summary_path = (
            RESULT_DIR
            / (
                "Temporal_Immune_Localizer_training_summary.json"
                if len(relation_results) == 1
                else "relation_localizer_suite_training_summary.json"
            )
        )

        save_json(
            {
                "dataset_path": str(DATASET_PATH),
                "dataset_profile": args.dataset_profile,
                "device": str(device),
                "configuration": {
                    "random_seed": RANDOM_SEED,
                    "relation_epochs": args.relation_epochs,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "hidden_channels": HIDDEN_CHANNELS,
                    "latent_channels": LATENT_CHANNELS,
                    "dropout": DROPOUT,
                    "relation_mask_start_fraction": (
                        RELATION_MASK_START_FRACTION
                    ),
                    "relation_mask_end_fraction": (
                        RELATION_MASK_END_FRACTION
                    ),
                "relation_mask_curriculum_epochs": (
                    RELATION_MASK_CURRICULUM_EPOCHS
                ),
                "relation_causal_pair_memory": True,
                "relation_persistent_edge_score": (
                    "one_step_ahead_absolute_surprise"
                ),
                "relation_full_mask_probability": (
                    RELATION_FULL_MASK_PROBABILITY
                ),
                    "relation_surprise_mode": (
                        RELATION_SURPRISE_MODE
                    ),
                    "relation_positive_surprise_weight": (
                        RELATION_POSITIVE_SURPRISE_WEIGHT
                    ),
                    "relation_absolute_surprise_weight": (
                        RELATION_ABSOLUTE_SURPRISE_WEIGHT
                    ),
                    "relation_inference_mask_groups": (
                        RELATION_INFERENCE_MASK_GROUPS
                    ),
                    "relation_causal_pair_memory": True,
                    "relation_persistent_edge_score": (
                        "one_step_ahead_absolute_surprise"
                    ),
                    "relation_min_epochs_before_stop": (
                        RELATION_MIN_EPOCHS_BEFORE_STOP
                    ),
                    "relation_early_stopping_patience": (
                        RELATION_EARLY_STOPPING_PATIENCE
                    ),
                    "relation_masked_edge_weight": RELATION_MASKED_EDGE_WEIGHT,
                    "relation_next_edge_weight": RELATION_NEXT_EDGE_WEIGHT,
                    "relation_memoryless_aux_weight": (
                        RELATION_MEMORYLESS_AUX_WEIGHT
                    ),
                    "relation_compactness_weight": RELATION_COMPACTNESS_WEIGHT,
                    "relation_topology_weight": RELATION_TOPOLOGY_WEIGHT,
                },
                "dataset": {
                    "train_normal_graphs": len(train_normal),
                    "val_normal_graphs": len(val_normal),
                    "in_channels": in_channels,
                    "edge_attr_dim": edge_attr_dim,
                    "node_normalization_fit_split": (
                        node_normalization[
                            "fit_split"
                        ]
                    ),
                    "edge_normalization_fit_split": (
                        edge_normalization[
                            "fit_split"
                        ]
                    ),
                },
                "temporal_sequence_training": temporal_sequence_summary,
                "models": [
                    result["summary"]
                    for result in relation_results
                ],
            },
            relation_summary_path,
        )

        print("\n" + "=" * 72)
        print("🎉 Relation localizer suite 训练完成")
        print("=" * 72)
        print(
            f"训练摘要：{relation_summary_path}"
        )
        print(
            "训练记录："
        )

        for relation_model_name in relation_model_names:
            print(
                f"  - {display_model_name(relation_model_name)}: "
                f"{model_history_path(relation_model_name)}"
            )

        return

    train_generator = torch.Generator(
        device="cpu"
    )

    train_generator.manual_seed(
        RANDOM_SEED
    )

    train_loader = DataLoader(
        train_normal,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=train_generator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_normal,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        "\n--- 2. 初始化并训练三个对比模型 ---"
    )

    all_summaries = []
    all_history_rows = []

    for model_name in MODEL_NAMES:
        config = ModelConfig(
            model_name=model_name,
            in_channels=in_channels,
            edge_attr_dim=edge_attr_dim,
        )

        result = train_model(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            dataset_meta=dataset_meta,
            node_normalization=(
                node_normalization
            ),
        )

        all_summaries.append(
            result["summary"]
        )

        all_history_rows.extend(
            result["history"]
        )

    print(
        "\n--- 3. 构建 normal sequence windows 并训练 Temporal_Immune_GAE ---"
    )

    (
        temporal_train_sequences,
        temporal_val_sequences,
        temporal_sequence_summary,
    ) = build_temporal_training_sequences(package)

    print(
        f"✅ temporal train sequences: "
        f"{len(temporal_train_sequences)}"
    )

    print(
        f"✅ temporal val sequences: "
        f"{len(temporal_val_sequences)}"
    )

    temporal_config = ModelConfig(
        model_name="Temporal_Immune_GAE",
        in_channels=in_channels,
        edge_attr_dim=edge_attr_dim,
    )

    temporal_result = train_temporal_model(
        config=temporal_config,
        train_sequences=temporal_train_sequences,
        val_sequences=temporal_val_sequences,
        device=device,
        dataset_meta=dataset_meta,
        node_normalization=node_normalization,
        sequence_summary=temporal_sequence_summary,
    )

    all_summaries.append(
        temporal_result["summary"]
    )

    all_history_rows.extend(
        temporal_result["history"]
    )

    print(
        "\n--- 4. 训练 TRAIL 及其显式消融 ---"
    )

    for relation_model_name in [
        RELATION_GCN_ABLATION_NAME,
        RELATION_EDGE_MEMORY_ABLATION_NAME,
        RELATION_MODEL_NAME,
    ]:
        relation_result = train_relation_localizer_model(
            config=ModelConfig(
                model_name=relation_model_name,
                in_channels=in_channels,
                edge_attr_dim=edge_attr_dim,
            ),
            train_sequences=temporal_train_sequences,
            val_sequences=temporal_val_sequences,
            device=device,
            dataset_meta=dataset_meta,
            node_normalization=node_normalization,
            sequence_summary=temporal_sequence_summary,
            epochs=RELATION_LOCALIZER_EPOCHS,
        )

        all_summaries.append(
            relation_result["summary"]
        )

        all_history_rows.extend(
            relation_result["history"]
        )

    save_history_csv(
        all_history_rows,
        COMBINED_HISTORY_PATH,
    )

    training_summary = {
        "dataset_path": str(
            DATASET_PATH
        ),
        "dataset_profile": args.dataset_profile,
        "device": str(device),
        "configuration": {
            "random_seed": RANDOM_SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": (
                LEARNING_RATE
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "early_stopping_patience": (
                EARLY_STOPPING_PATIENCE
            ),
            "hidden_channels": (
                HIDDEN_CHANNELS
            ),
            "latent_channels": (
                LATENT_CHANNELS
            ),
            "dropout": DROPOUT,
            "negative_ratio": (
                NEGATIVE_RATIO
            ),
            "topology_weight": (
                TOPOLOGY_WEIGHT
            ),
            "edge_attribute_weight": (
                EDGE_ATTRIBUTE_WEIGHT
            ),
            "node_reconstruction_weight": (
                NODE_RECONSTRUCTION_WEIGHT
            ),
            "latent_regularization_weight": (
                LATENT_REGULARIZATION_WEIGHT
            ),
            "positive_topology_edge_weighting": "uniform",
            "temporal_epochs": (
                TEMPORAL_EPOCHS
            ),
            "temporal_sequence_length": (
                TEMPORAL_SEQUENCE_LENGTH
            ),
            "temporal_sequence_stride": (
                TEMPORAL_SEQUENCE_STRIDE
            ),
            "temporal_train_on_full_clips": (
                TEMPORAL_TRAIN_ON_FULL_CLIPS
            ),
            "relation_localizer_epochs": (
                RELATION_LOCALIZER_EPOCHS
            ),
            "relation_mask_curriculum": {
                "start_fraction": (
                    RELATION_MASK_START_FRACTION
                ),
                "end_fraction": (
                    RELATION_MASK_END_FRACTION
                ),
                "curriculum_epochs": (
                    RELATION_MASK_CURRICULUM_EPOCHS
                ),
                "full_mask_probability": (
                    RELATION_FULL_MASK_PROBABILITY
                ),
            },
            "relation_surprise_mode": (
                RELATION_SURPRISE_MODE
            ),
            "relation_inference_mask_groups": (
                RELATION_INFERENCE_MASK_GROUPS
            ),
            "relation_min_epochs_before_stop": (
                RELATION_MIN_EPOCHS_BEFORE_STOP
            ),
            "relation_early_stopping_patience": (
                RELATION_EARLY_STOPPING_PATIENCE
            ),
        },
        "dataset": {
            "train_normal_graphs": len(
                train_normal
            ),
            "val_normal_graphs": len(
                val_normal
            ),
            "in_channels": (
                in_channels
            ),
            "edge_attr_dim": (
                edge_attr_dim
            ),
            "node_normalization_fit_split": (
                node_normalization[
                    "fit_split"
                ]
            ),
        },
        "temporal_sequence_training": (
            temporal_sequence_summary
        ),
        "models": all_summaries,
        "combined_history_path": str(
            COMBINED_HISTORY_PATH
        ),
    }

    save_json(
        training_summary,
        SUMMARY_PATH,
    )

    print("\n" + "=" * 72)
    print(
        "🎉 Baselines、ablations 和 "
        "TRAIL 训练完成"
    )
    print("=" * 72)

    for summary in all_summaries:
        print(
            f"{summary['model_name']:<20} | "
            f"BestEpoch={summary['best_epoch']:>3} | "
            f"ValLoss={summary['best_val_loss']:.6f} | "
            f"Params={summary['parameter_count']:,}"
        )

    print(
        f"\n训练摘要：{SUMMARY_PATH}"
    )

    print(
        f"总训练记录："
        f"{COMBINED_HISTORY_PATH}"
    )

    print(
        "\n下一步运行新的 03_evaluate_and_plot.py，"
        "评估峰值帧 AUROC、连续检测延迟、持续性和边级定位。"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断运行。")
        sys.exit(130)
    except Exception as error:
        print(
            f"\n❌ 02 训练失败：{error}"
        )
        raise
