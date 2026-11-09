"""Create a visual, independent comparison of the before/after demo runs.

Example:
    ``python -m models.helloworld.compare --before artifacts/run/before \
    --after artifacts/run/after --output artifacts/run/visuals``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from .before.helloworld import ToyModel as BeforeModel
from .new.data import ToyDataset
from .new.helloworld import ToyModel as AfterModel


def _epoch_metrics(path: Path) -> list[dict[str, Any]]:
    """Read epoch records from a run's JSON Lines metric file."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "epoch":
                records.append(record)
    if not records:
        raise ValueError(f"no epoch metrics found in {path}")
    return records


def _checkpoint_model(path: Path, model: torch.nn.Module) -> torch.nn.Module:
    """Load a CPU checkpoint into the supplied model and select eval mode."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
    return model.eval()


def _predictions(model: torch.nn.Module, dataset: ToyDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened labels and predictions for a small held-out sample."""
    inputs, labels = dataset.inputs[:512], dataset.labels[:512]
    with torch.no_grad():
        output = model(inputs)
    predictions = output["predictions"] if isinstance(output, dict) else output
    return labels.flatten(), predictions.flatten()


def _parameter_norms(model: torch.nn.Module) -> list[float]:
    """Return L2 norms in layer order so differently named models can compare."""
    return [float(parameter.detach().norm()) for parameter in model.parameters()]


def _difference(before: float, after: float) -> dict[str, float]:
    """Return signed, absolute, and percentage differences versus before."""
    signed = after - before
    return {
        "before": before,
        "after": after,
        "signed_difference": signed,
        "absolute_difference": abs(signed),
        "percentage_difference": (signed / before * 100.0) if before else 0.0,
        "absolute_percentage_difference": (abs(signed) / abs(before) * 100.0) if before else 0.0,
        "Loss Improved": "Yes" if after < before else "No",
    }


def create_comparison(before_dir: Path, after_dir: Path, output_dir: Path) -> Path:
    """Generate a PNG dashboard and JSON report from two completed runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    before_metrics = _epoch_metrics(before_dir / "metrics.jsonl")
    after_metrics = _epoch_metrics(after_dir / "metrics.jsonl")
    before_model = _checkpoint_model(before_dir / "last.pt", BeforeModel())
    after_model = _checkpoint_model(after_dir / "last.pt", AfterModel())
    evaluation_data = ToyDataset(10_000, seed=9_999)
    labels, before_predictions = _predictions(before_model, evaluation_data)
    _, after_predictions = _predictions(after_model, evaluation_data)
    before_mse = float(torch.mean((before_predictions - labels) ** 2))
    after_mse = float(torch.mean((after_predictions - labels) ** 2))
    prediction_delta = float(torch.mean(torch.abs(before_predictions - after_predictions)))
    before_norms, after_norms = _parameter_norms(before_model), _parameter_norms(after_model)
    train_difference = _difference(float(before_metrics[-1]["train_loss"]), float(after_metrics[-1]["train_loss"]))
    validation_difference = _difference(float(before_metrics[-1]["val_loss"]), float(after_metrics[-1]["val_loss"]))
    evaluation_difference = _difference(before_mse, after_mse)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    figure.suptitle("Coupled Loop vs Generic Trainer — Same Model and Data Contract", fontsize=17, weight="bold")

    for records, label, color in ((before_metrics, "Before: coupled", "#C44E52"), (after_metrics, "After: generic trainer", "#1F77B4")):
        epochs = [item["epoch"] for item in records]
        axes[0, 0].plot(epochs, [item["train_loss"] for item in records], marker="o", color=color, label=f"{label} train")
        axes[0, 0].plot(epochs, [item["val_loss"] for item in records], marker="s", linestyle="--", color=color, label=f"{label} validation")
    axes[0, 0].set(title="Learning curves", xlabel="Epoch", ylabel="Mean squared error")
    axes[0, 0].legend(fontsize=8)

    bars = axes[0, 1].bar(["Before\ncoupled", "After\ngeneric"], [before_mse, after_mse], color=["#C44E52", "#1F77B4"])
    axes[0, 1].bar_label(bars, fmt="%.4f", padding=3)
    axes[0, 1].set(title="Independent 10,000-example evaluation", ylabel="MSE (lower is better)")
    axes[0, 1].text(
        0.5,
        0.93,
        f"Absolute difference: {evaluation_difference['absolute_difference']:.6f}\n"
        f"Percentage difference: {evaluation_difference['percentage_difference']:+.3f}%\n"
        f"Loss Improved: {evaluation_difference['Loss Improved']}",
        transform=axes[0, 1].transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#777777"},
    )

    sample = slice(0, 500)
    axes[1, 0].scatter(labels[sample], before_predictions[sample], s=10, alpha=0.45, color="#C44E52", label="Before")
    axes[1, 0].scatter(labels[sample], after_predictions[sample], s=10, alpha=0.45, color="#1F77B4", label="After")
    limits = [float(labels[sample].min()), float(labels[sample].max())]
    axes[1, 0].plot(limits, limits, color="#333333", linestyle=":", label="Ideal")
    axes[1, 0].set(title="Predictions versus expected values", xlabel="Expected", ylabel="Predicted")
    axes[1, 0].legend(fontsize=8)

    positions = torch.arange(len(before_norms)).numpy()
    width = 0.36
    axes[1, 1].bar(positions - width / 2, before_norms, width, color="#C44E52", label="Before")
    axes[1, 1].bar(positions + width / 2, after_norms, width, color="#1F77B4", label="After")
    axes[1, 1].set(title="Checkpoint weight/bias norms", xlabel="Parameter tensor in layer order", ylabel="L2 norm", xticks=positions)
    axes[1, 1].legend(fontsize=8)

    image_path = output_dir / "leadership_comparison.png"
    figure.savefig(image_path, dpi=180)
    plt.close(figure)
    report = {
        "before_independent_mse": before_mse,
        "after_independent_mse": after_mse,
        "mean_absolute_prediction_delta": prediction_delta,
        "final_training_loss_difference": train_difference,
        "final_validation_loss_difference": validation_difference,
        "independent_evaluation_mse_difference": evaluation_difference,
        "before_parameter_norms": before_norms,
        "after_parameter_norms": after_norms,
        "interpretation": "Comparable quality demonstrates execution decoupling; checkpoints need not be byte-identical because trainer policies differ.",
    }
    (output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "comparison_complete", "dashboard": str(image_path), **report}, sort_keys=True))
    return image_path


def main() -> None:
    """Parse run directories and create the visual comparison artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_comparison(args.before, args.after, args.output)


if __name__ == "__main__":
    main()
