"""Clean family-disjoint RiskBench retraining of the Social-GAN architecture.

This stage trains only on safe-core trajectories and selects a checkpoint on a
family-disjoint safe-core subset.  It is deliberately separate from event-risk
evaluation: the first run is a one-fold feasibility check that verifies data,
device compatibility, optimization, and held-out trajectory error before any
outer danger-event scores are opened.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .mainstream_baseline_evaluation import _load_state
from .mainstream_baseline_evaluation_v2 import (
    DEFAULT_INDEX_ROOT,
    _family_dev_split,
    _load_index,
)
from .vendor_social_gan.losses import gan_d_loss, gan_g_loss, l2_loss
from .vendor_social_gan.models import TrajectoryDiscriminator, TrajectoryGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/riskbench_social_gan_clean_retrain_v1"
PROTOCOL = "riskbench-social-gan-safe-only-clean-retrain-v1"
OBS_LEN = 20
PRED_LEN = 30
TOTAL_LEN = OBS_LEN + PRED_LEN
MAX_ACTORS = 8
WINDOW_STRIDE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--best-k", type=int, default=10)
    parser.add_argument("--validation-samples", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def relative_trajectory(absolute: torch.Tensor) -> torch.Tensor:
    relative = torch.zeros_like(absolute)
    relative[1:] = absolute[1:] - absolute[:-1]
    return relative


def relative_to_absolute(relative: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(relative, dim=0) + start.unsqueeze(0)


def _complete_actor_ids(
    state: dict[int, dict[str, np.ndarray]], frames: Sequence[int]
) -> list[str]:
    if not frames:
        return []
    common = set(str(actor) for actor in state[frames[0]])
    for frame in frames[1:]:
        common.intersection_update(str(actor) for actor in state[frame])
    endpoint = frames[OBS_LEN - 1]
    return sorted(
        common,
        key=lambda actor: (
            float(np.linalg.norm(state[endpoint][actor])),
            actor,
        ),
    )[:MAX_ACTORS]


def materialize_scenes(records: Sequence[dict[str, Any]]) -> list[np.ndarray]:
    """Create complete 20-observation/30-target social scenes.

    Coordinates are already ego relative.  A stationary origin trajectory is
    included as the ego actor, so generated road-user paths can later be scored
    relative to a jointly predicted ego reference without using the true future.
    """
    scenes: list[np.ndarray] = []
    for record in records:
        state = _load_state(record["state_path"])
        available = sorted(state)
        if len(available) < TOTAL_LEN:
            continue
        available_set = set(available)
        for start in range(available[0], available[-1] - TOTAL_LEN + 2, WINDOW_STRIDE):
            frames = list(range(start, start + TOTAL_LEN))
            if any(frame not in available_set for frame in frames):
                continue
            actors = _complete_actor_ids(state, frames)
            if not actors:
                continue
            values = np.zeros((TOTAL_LEN, 1 + len(actors), 2), dtype=np.float32)
            for actor_index, actor in enumerate(actors, start=1):
                values[:, actor_index] = np.stack(
                    [np.asarray(state[frame][actor], dtype=np.float32) for frame in frames]
                )
            scenes.append(values)
    return scenes


class SceneDataset(Dataset):
    def __init__(self, scenes: Sequence[np.ndarray]):
        self.scenes = list(scenes)

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.scenes[index])


def collate_scenes(batch: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    starts: list[list[int]] = []
    cursor = 0
    for scene in batch:
        starts.append([cursor, cursor + scene.shape[1]])
        cursor += scene.shape[1]
    return torch.cat(list(batch), dim=1), torch.tensor(starts, dtype=torch.long)


def build_models(device: torch.device) -> tuple[TrajectoryGenerator, TrajectoryDiscriminator]:
    # These compact hyperparameters match the released RiskBench Social-GAN
    # checkpoint configuration; weights are retrained here from scratch.
    generator = TrajectoryGenerator(
        obs_len=OBS_LEN,
        pred_len=PRED_LEN,
        embedding_dim=16,
        encoder_h_dim=32,
        decoder_h_dim=32,
        mlp_dim=64,
        num_layers=1,
        noise_dim=(8,),
        noise_type="gaussian",
        noise_mix_type="global",
        pooling_type="pool_net",
        pool_every_timestep=False,
        dropout=0.0,
        bottleneck_dim=32,
        batch_norm=False,
    ).to(device)
    discriminator = TrajectoryDiscriminator(
        obs_len=OBS_LEN,
        pred_len=PRED_LEN,
        embedding_dim=16,
        h_dim=32,
        mlp_dim=64,
        num_layers=1,
        dropout=0.0,
        batch_norm=False,
        d_type="local",
    ).to(device)
    return generator, discriminator


def _batch_parts(
    trajectories: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    trajectories = trajectories.to(device)
    relative = relative_trajectory(trajectories)
    return (
        trajectories[:OBS_LEN],
        trajectories[OBS_LEN:],
        relative[:OBS_LEN],
        relative[OBS_LEN:],
    )


def discriminator_step(
    generator: TrajectoryGenerator,
    discriminator: TrajectoryDiscriminator,
    optimizer: torch.optim.Optimizer,
    trajectories: torch.Tensor,
    seq_start_end: torch.Tensor,
    device: torch.device,
) -> float:
    obs, future, obs_rel, future_rel = _batch_parts(trajectories, device)
    seq_start_end = seq_start_end.to(device)
    with torch.no_grad():
        fake_rel = generator(obs, obs_rel, seq_start_end)
        fake = relative_to_absolute(fake_rel, obs[-1])
    real_score = discriminator(
        torch.cat([obs, future]), torch.cat([obs_rel, future_rel]), seq_start_end
    )
    fake_score = discriminator(
        torch.cat([obs, fake]), torch.cat([obs_rel, fake_rel]), seq_start_end
    )
    loss = gan_d_loss(real_score, fake_score)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)
    optimizer.step()
    return float(loss.detach())


def generator_step(
    generator: TrajectoryGenerator,
    discriminator: TrajectoryDiscriminator,
    optimizer: torch.optim.Optimizer,
    trajectories: torch.Tensor,
    seq_start_end: torch.Tensor,
    device: torch.device,
    best_k: int,
) -> tuple[float, float, float]:
    obs, future, obs_rel, future_rel = _batch_parts(trajectories, device)
    seq_start_end = seq_start_end.to(device)
    masks = torch.ones((obs.shape[1], PRED_LEN), device=device)
    predictions: list[torch.Tensor] = []
    raw_losses: list[torch.Tensor] = []
    for _ in range(best_k):
        prediction = generator(obs, obs_rel, seq_start_end)
        predictions.append(prediction)
        raw_losses.append(l2_loss(prediction, future_rel, masks, mode="raw"))
    raw = torch.stack(raw_losses, dim=1)
    reconstruction = obs.new_zeros(())
    for start, end in seq_start_end:
        start_i, end_i = int(start), int(end)
        reconstruction = reconstruction + raw[start_i:end_i].sum(dim=0).min()
    reconstruction = reconstruction / masks.sum()
    chosen = predictions[-1]
    fake = relative_to_absolute(chosen, obs[-1])
    adversarial = gan_g_loss(
        discriminator(
            torch.cat([obs, fake]), torch.cat([obs_rel, chosen]), seq_start_end
        )
    )
    loss = reconstruction + adversarial
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
    optimizer.step()
    return float(loss.detach()), float(reconstruction.detach()), float(adversarial.detach())


@torch.no_grad()
def validation_ade(
    generator: TrajectoryGenerator,
    loader: DataLoader,
    device: torch.device,
    samples: int,
) -> float:
    generator.eval()
    total_error = 0.0
    total_points = 0
    for trajectories, seq_start_end in loader:
        obs, future, obs_rel, _ = _batch_parts(trajectories, device)
        seq_start_end = seq_start_end.to(device)
        errors = []
        for _ in range(samples):
            predicted_rel = generator(obs, obs_rel, seq_start_end)
            predicted = relative_to_absolute(predicted_rel, obs[-1])
            errors.append(torch.linalg.norm(predicted - future, dim=-1).mean(dim=0))
        best = torch.stack(errors).min(dim=0).values
        total_error += float(best.sum())
        total_points += int(best.numel())
    generator.train()
    return total_error / max(total_points, 1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(args.device)
    index = _load_index(args.index_root.expanduser().resolve(), args.fold)
    train_records, selection_records = _family_dev_split(index["splits"]["train_normal"])
    train_scenes = materialize_scenes(train_records)
    selection_scenes = materialize_scenes(selection_records)
    if args.smoke:
        train_scenes = train_scenes[:16]
        selection_scenes = selection_scenes[:8]
        args.epochs = min(args.epochs, 2)
        args.best_k = min(args.best_k, 2)
        args.validation_samples = min(args.validation_samples, 2)
    if not train_scenes or not selection_scenes:
        raise RuntimeError("No complete Social-GAN scenes were materialized.")
    train_loader = DataLoader(
        SceneDataset(train_scenes), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_scenes, num_workers=0,
    )
    selection_loader = DataLoader(
        SceneDataset(selection_scenes), batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_scenes, num_workers=0,
    )
    generator, discriminator = build_models(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=5e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)
    history: list[dict[str, float | int]] = []
    best_ade = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        d_values: list[float] = []
        g_values: list[float] = []
        rec_values: list[float] = []
        adv_values: list[float] = []
        for trajectories, seq_start_end in train_loader:
            d_values.append(discriminator_step(
                generator, discriminator, optimizer_d, trajectories,
                seq_start_end, device,
            ))
            g, rec, adv = generator_step(
                generator, discriminator, optimizer_g, trajectories,
                seq_start_end, device, args.best_k,
            )
            g_values.append(g); rec_values.append(rec); adv_values.append(adv)
        ade = validation_ade(generator, selection_loader, device, args.validation_samples)
        row = {
            "epoch": epoch,
            "discriminator_loss": float(np.mean(d_values)),
            "generator_loss": float(np.mean(g_values)),
            "reconstruction_loss": float(np.mean(rec_values)),
            "adversarial_loss": float(np.mean(adv_values)),
            "selection_min_ade_m": ade,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if np.isfinite(ade) and ade < best_ade - 1e-5:
            best_ade = ade
            best_epoch = epoch
            best_state = copy.deepcopy(generator.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Training produced no finite selection checkpoint.")
    output = args.output_root.expanduser().resolve() / f"fold_{args.fold:02d}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "generator_safe_selection_best.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "fold": args.fold,
            "seed": args.seed,
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "selection_min_ade_m": best_ade,
            "architecture": {
                "obs_len": OBS_LEN, "pred_len": PRED_LEN,
                "embedding_dim": 16, "hidden_dim": 32, "mlp_dim": 64,
                "noise_dim": 8, "pooling": "pool_net",
            },
        }, checkpoint,
    )
    result = {
        "protocol": PROTOCOL,
        "status": "one_fold_feasibility_complete",
        "fold": args.fold,
        "claim_tier": "published_architecture_cleanly_retrained_on_safe_core",
        "outer_danger_events_opened": False,
        "positive_labels_used": False,
        "train_family_count": len({r["scenario_group_id"] for r in train_records}),
        "selection_family_count": len({r["scenario_group_id"] for r in selection_records}),
        "train_scene_count": len(train_scenes),
        "selection_scene_count": len(selection_scenes),
        "best_epoch": best_epoch,
        "selection_min_ade_m": best_ade,
        "checkpoint": str(checkpoint),
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
    }
    (output / "training_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
