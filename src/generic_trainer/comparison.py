"""Generic visual comparison for any two Generic Trainer artifact directories.

Each directory must contain ``metrics.jsonl`` and ``last.pt``. The utility
does not import model code, making it suitable for traditional models, LLMs,
CPU runs, and distributed runs that honor the standard artifact contract.

Example:
    ``python -m generic_trainer.comparison --before artifacts/baseline \
    --after artifacts/candidate --output artifacts/comparison``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


def loss_difference(before: float, after: float) -> dict[str, float | str]:
    """Describe an after loss relative to the before loss baseline."""
    signed = after - before
    percentage = signed / before * 100.0 if before else 0.0
    return {
        "before": before,
        "after": after,
        "absolute_difference": abs(signed),
        "percentage_difference": percentage,
        "Loss Improved": "Yes" if after < before else "No",
    }


def _epochs(directory: Path) -> list[dict[str, Any]]:
    """Load standardized epoch events from one artifact directory."""
    path = directory / "metrics.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    epochs = [record for record in records if record.get("event") == "epoch"]
    if not epochs:
        raise ValueError(f"no epoch events found in {path}")
    return epochs


def _tensor_norms(directory: Path) -> list[float]:
    """Load checkpoint tensors and return norms in serialized order."""
    checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    tensors = [value for value in state.values() if torch.is_tensor(value)]
    if not tensors:
        raise ValueError(f"no model tensors found in {directory / 'last.pt'}")
    return [float(value.detach().float().norm()) for value in tensors]


def compare_runs(
    before_dir: Path,
    after_dir: Path,
    output_dir: Path,
    before_label: str = "Before",
    after_label: str = "After",
) -> Path:
    """Create a model-independent loss and checkpoint comparison dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    before_epochs, after_epochs = _epochs(before_dir), _epochs(after_dir)
    train = loss_difference(float(before_epochs[-1]["train_loss"]), float(after_epochs[-1]["train_loss"]))
    validation = loss_difference(float(before_epochs[-1]["val_loss"]), float(after_epochs[-1]["val_loss"]))
    before_norms, after_norms = _tensor_norms(before_dir), _tensor_norms(after_dir)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    figure.suptitle("Training Run Comparison", fontsize=18, weight="bold")
    colors = ("#C44E52", "#1F77B4")
    for records, label, color in (
        (before_epochs, before_label, colors[0]),
        (after_epochs, after_label, colors[1]),
    ):
        x = [record["epoch"] for record in records]
        axes[0, 0].plot(x, [record["train_loss"] for record in records], marker="o", color=color, label=f"{label} train")
        axes[0, 0].plot(x, [record["val_loss"] for record in records], marker="s", linestyle="--", color=color, label=f"{label} validation")
    axes[0, 0].set(title="Learning curves", xlabel="Epoch", ylabel="Loss")
    axes[0, 0].legend(fontsize=8)

    bars = axes[0, 1].bar(
        [f"{before_label}\ntrain", f"{after_label}\ntrain", f"{before_label}\nvalidation", f"{after_label}\nvalidation"],
        [train["before"], train["after"], validation["before"], validation["after"]],
        color=[colors[0], colors[1], colors[0], colors[1]],
    )
    axes[0, 1].bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
    axes[0, 1].set(title="Final losses", ylabel="Loss lower is better")

    count = min(len(before_norms), len(after_norms), 20)
    positions = torch.arange(count).numpy()
    axes[1, 0].bar(positions - 0.18, before_norms[:count], 0.36, color=colors[0], label=before_label)
    axes[1, 0].bar(positions + 0.18, after_norms[:count], 0.36, color=colors[1], label=after_label)
    axes[1, 0].set(title="Checkpoint tensor norms first 20", xlabel="Tensor order", ylabel="L2 norm", xticks=positions)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].axis("off")
    table_rows = [
        ["Training", f"{train['absolute_difference']:.6f}", f"{train['percentage_difference']:+.3f}%", train["Loss Improved"]],
        ["Validation", f"{validation['absolute_difference']:.6f}", f"{validation['percentage_difference']:+.3f}%", validation["Loss Improved"]],
    ]
    table = axes[1, 1].table(
        cellText=table_rows,
        colLabels=["Metric", "Absolute difference", "Percentage difference", "Loss Improved"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.8)
    axes[1, 1].set_title("Leadership summary", pad=18)

    dashboard = output_dir / "run_comparison.png"
    figure.savefig(dashboard, dpi=180)
    plt.close(figure)
    report = {
        "before_directory": str(before_dir),
        "after_directory": str(after_dir),
        "final_training_loss_difference": train,
        "final_validation_loss_difference": validation,
        "checkpoint_tensors_compared": count,
    }
    (output_dir / "run_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "comparison_complete", "dashboard": str(dashboard), **report}, sort_keys=True))
    return dashboard


def main() -> None:
    """Parse generic artifact directories and generate comparison outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-label", default="Before")
    parser.add_argument("--after-label", default="After")
    args = parser.parse_args()
    compare_runs(args.before, args.after, args.output, args.before_label, args.after_label)


if __name__ == "__main__":
    main()
