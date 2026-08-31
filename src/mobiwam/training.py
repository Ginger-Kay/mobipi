from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor, nn

from .losses import OBCWAMLoss
from .models import (
    EvaluatorConfig,
    OBCWAM,
    TrajectoryOnlyEvaluator,
    ValueOnlyEvaluator,
    trainable_parameter_count,
)


MODEL_SEEDS = (17, 23, 41)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(name: str, config: EvaluatorConfig) -> nn.Module:
    if name == "obc-wam":
        return OBCWAM(config)
    if name == "value-only":
        return ValueOnlyEvaluator(config)
    if name == "trajectory-only":
        return TrajectoryOnlyEvaluator(config)
    raise ValueError(f"unknown model: {name}")


def _model_inputs(data: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict[str, Tensor]:
    return {
        "context": torch.as_tensor(data["context"][indices], device=device),
        "option_ids": torch.as_tensor(data["option_ids"][indices], device=device).long(),
        "candidate_params": torch.as_tensor(data["candidate_params"][indices], device=device),
        "phase_ids": torch.as_tensor(data["phase_ids"][indices], device=device).long(),
        "duration": torch.as_tensor(data["duration"][indices], device=device),
    }


def _generic_loss(predictions: dict[str, Tensor], data: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> Tensor:
    success = torch.as_tensor(data["success"][indices], device=device).reshape(-1, 1)
    risk = torch.as_tensor(data["irreversible_risk"][indices], device=device).reshape(-1, 1)
    cost = torch.as_tensor(data["duration_cost"][indices], device=device).reshape(-1, 6)
    return (
        nn.functional.binary_cross_entropy_with_logits(predictions["success"], success)
        + nn.functional.binary_cross_entropy_with_logits(predictions["irreversible_risk"], risk)
        + nn.functional.smooth_l1_loss(predictions["duration_cost"], cost)
    )


def source_group_route_regret(
    source_ids: np.ndarray,
    success_logits: np.ndarray,
    risk_logits: np.ndarray,
    realized_success: np.ndarray,
    realized_risk: np.ndarray,
) -> float:
    regrets = []
    for source_id in np.unique(source_ids):
        indices = np.flatnonzero(source_ids == source_id)
        predicted_utility = success_logits[indices] - risk_logits[indices]
        realized_utility = realized_success[indices] - realized_risk[indices]
        selected = int(np.argmax(predicted_utility))
        regrets.append(float(np.max(realized_utility) - realized_utility[selected]))
    return float(np.mean(regrets)) if regrets else float("inf")


def _group_batches(
    indices: np.ndarray,
    source_ids: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[np.ndarray]:
    groups = [indices[source_ids[indices] == source_id] for source_id in np.unique(source_ids[indices])]
    rng.shuffle(groups)
    pending: list[np.ndarray] = []
    pending_rows = 0
    for group in groups:
        if pending and pending_rows + len(group) > batch_size:
            yield np.concatenate(pending)
            pending = []
            pending_rows = 0
        pending.append(group)
        pending_rows += len(group)
    if pending:
        yield np.concatenate(pending)


def _structured_targets(
    data: dict[str, np.ndarray], indices: np.ndarray, device: torch.device
) -> dict[str, Tensor]:
    source_ids = data["source_ids"][indices]
    pair_indices = []
    pair_preferred = []
    a0_indices = []
    e_indices = []
    for source_id in np.unique(source_ids):
        local = np.flatnonzero(source_ids == source_id)
        utility = (
            data["success"][indices][local].reshape(-1)
            - data["irreversible_risk"][indices][local].reshape(-1)
            - 0.001 * data["duration_cost"][indices][local, 0]
        )
        pair_indices.append([int(local[np.argmax(utility)]), int(local[np.argmin(utility)])])
        pair_preferred.append(1.0)
        a0 = local[data["is_a0"][indices][local].astype(bool)]
        execute = local[data["option_ids"][indices][local] == 0]
        if not len(a0) or not len(execute):
            raise ValueError(f"source {source_id} lacks paired A(0)/E supervision")
        a0_indices.append(int(a0[0]))
        e_indices.append(int(execute[0]))
    return {
        "typed_internal_states": torch.as_tensor(data["typed_internal_states"][indices], device=device),
        "common_boundary_latent": torch.as_tensor(data["common_boundary_latent"][indices], device=device),
        "success": torch.as_tensor(data["success"][indices], device=device).reshape(-1, 1),
        "irreversible_risk": torch.as_tensor(data["irreversible_risk"][indices], device=device).reshape(-1, 1),
        "duration_cost": torch.as_tensor(data["duration_cost"][indices], device=device).reshape(-1, 6),
        "pair_indices": torch.as_tensor(pair_indices, device=device),
        "pair_preferred": torch.as_tensor(pair_preferred, device=device),
        "a0_indices": torch.as_tensor(a0_indices, device=device),
        "e_indices": torch.as_tensor(e_indices, device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a matched MobiWAM evaluator")
    parser.add_argument("--model", choices=["value-only", "trajectory-only", "obc-wam"], required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=MODEL_SEEDS, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--candidate-dim", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_epochs > 50 or args.patience != 8:
        parser.error("MMWAM-OBC-001 freezes max_epochs<=50 and patience=8")
    if args.batch_size <= 0 or args.gradient_accumulation <= 0:
        parser.error("batch size and gradient accumulation must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    data_file = np.load(args.features, allow_pickle=False)
    data = {name: data_file[name] for name in data_file.files}
    required = {
        "context", "option_ids", "candidate_params", "phase_ids", "duration",
        "success", "irreversible_risk", "duration_cost", "source_ids", "split",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"feature file is missing: {missing}")
    train_indices = np.flatnonzero(data["split"].astype(str) == "train")
    validation_indices = np.flatnonzero(data["split"].astype(str) == "validation")
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("training and validation source groups must both be non-empty")

    config = EvaluatorConfig(input_dim=args.input_dim, candidate_dim=args.candidate_dim)
    model = build_model(args.model, config).to(device)
    parameter_count = trainable_parameter_count(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    rng = np.random.default_rng(args.seed)
    best_regret = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    args.output_root.mkdir(parents=True, exist_ok=False)
    history = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for epoch in range(args.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        batch_count = 0
        for batch_count, indices in enumerate(
            _group_batches(train_indices, data["source_ids"], args.batch_size, rng),
            start=1,
        ):
            inputs = _model_inputs(data, indices, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                predictions = model(**inputs)
                loss = _generic_loss(predictions, data, indices, device)
                if args.model == "obc-wam":
                    structured_required = {
                        "typed_internal_states", "common_boundary_latent", "is_a0",
                    }
                    if not structured_required.issubset(data):
                        raise ValueError("OBC-WAM feature file lacks frozen structured targets")
                    targets = _structured_targets(data, indices, device)
                    loss = OBCWAMLoss()(predictions, targets)["loss"]
                loss = loss / args.gradient_accumulation
            loss.backward()
            if batch_count % args.gradient_accumulation == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_loss += float(loss.detach()) * args.gradient_accumulation
        if batch_count % args.gradient_accumulation:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            predictions = model(**_model_inputs(data, validation_indices, device))
        success_logits = predictions["success"].float().cpu().numpy().reshape(-1)
        risk_logits = predictions["irreversible_risk"].float().cpu().numpy().reshape(-1)
        regret = source_group_route_regret(
            data["source_ids"][validation_indices], success_logits, risk_logits,
            data["success"][validation_indices].reshape(-1),
            data["irreversible_risk"][validation_indices].reshape(-1),
        )
        history.append({"epoch": epoch, "train_loss": train_loss / max(batch_count, 1), "validation_source_group_route_regret": regret})
        if regret < best_regret:
            best_regret = regret
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "config": config.__dict__, "model_name": args.model, "seed": args.seed, "epoch": epoch}, args.output_root / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record = {
        "status": "completed", "model": args.model, "seed": args.seed,
        "started_at": started_at, "ended_at": ended_at,
        "features": str(args.features.resolve()), "features_sha256": sha256_file(args.features),
        "trainable_parameters": parameter_count, "best_epoch": best_epoch,
        "best_validation_source_group_route_regret": best_regret,
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 0.05, "gradient_clip": 1.0, "schedule": "cosine"},
        "bf16": device.type == "cuda", "history": history,
    }
    (args.output_root / "training.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
