from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .models import EvaluatorConfig
from .training import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure batched evaluator and end-to-end latency")
    parser.add_argument("--model", choices=["value-only", "trajectory-only", "obc-wam"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--proposer-ms", type=float, required=True)
    parser.add_argument("--encoder-ms", type=float, required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    config = EvaluatorConfig(**payload["config"])
    model = build_model(args.model, config).cuda().eval()
    model.load_state_dict(payload["model"])
    inputs = {
        "context": torch.randn(args.batch_size, 4, config.input_dim, device="cuda"),
        "option_ids": torch.arange(args.batch_size, device="cuda") % config.num_options,
        "candidate_params": torch.randn(args.batch_size, config.candidate_dim, device="cuda"),
        "phase_ids": torch.zeros(args.batch_size, dtype=torch.long, device="cuda"),
        "duration": torch.zeros(args.batch_size, device="cuda"),
    }
    with torch.no_grad():
        for _ in range(10):
            model(**inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            model(**inputs)
        torch.cuda.synchronize()
    evaluator_ms = (time.perf_counter() - start) * 1000.0 / args.iterations
    record = {
        "batch_size": args.batch_size, "iterations": args.iterations,
        "proposer_ms": args.proposer_ms, "encoder_ms": args.encoder_ms,
        "evaluator_ms": evaluator_ms,
        "total_ms": args.proposer_ms + args.encoder_ms + evaluator_ms,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite latency record: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
