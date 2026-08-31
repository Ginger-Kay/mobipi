from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_tensor = torch.as_tensor(logits, dtype=torch.float64).reshape(-1)
    labels_tensor = torch.as_tensor(labels, dtype=torch.float64).reshape(-1)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze calibration-only temperatures and operating point")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-irreversible", type=float, default=0.05)
    args = parser.parse_args()
    data_file = np.load(args.predictions, allow_pickle=False)
    data = {name: data_file[name] for name in data_file.files}
    if set(data["split"].astype(str)) != {"calibration"}:
        raise ValueError("calibration entry point refuses validation or locked-test rows")
    success_temperature = fit_temperature(data["success_logits"], data["success"])
    risk_temperature = fit_temperature(data["risk_logits"], data["irreversible_risk"])
    success_probability = 1.0 / (1.0 + np.exp(-data["success_logits"] / success_temperature))
    risk_probability = 1.0 / (1.0 + np.exp(-data["risk_logits"] / risk_temperature))
    eligible = risk_probability <= args.max_irreversible
    tau_success = float(np.quantile(success_probability[eligible], 0.25)) if np.any(eligible) else 1.0
    record = {
        "split": "calibration", "success_temperature": success_temperature,
        "risk_temperature": risk_temperature, "tau_success": tau_success,
        "epsilon_risk": args.max_irreversible,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
