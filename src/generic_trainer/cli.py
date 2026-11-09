"""Command-line orchestration for configuration-driven training.

Examples:
    Run a local training job::

        python -m generic_trainer.cli --config configs/smoke.yaml

    Override an existing YAML value::

        python -m generic_trainer.cli --config configs/smoke.yaml \
            --set trainer.max_epochs=2
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable

import yaml
import torch

from .config import load_config
from .distributed import initialize, shutdown
from .engine import Trainer
from .environment import apply_environment
from .preflight import validate_capacity
from .telemetry import model_size


def import_factory(path: str) -> Callable[..., Any]:
    """Import a callable expressed as ``package.module:function``.

    Args:
        path: Fully qualified factory reference.

    Returns:
        The imported callable.

    Raises:
        ValueError: If ``path`` does not use ``module:function`` syntax.
        TypeError: If the referenced attribute is not callable.

    Example:
        ``import_factory("models.helloworld.new.factory:build_model")``
        returns the hello-world model factory.
    """
    module_name, separator, attribute = path.partition(":")
    if not separator:
        raise ValueError(f"factory must use module:function syntax: {path}")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"factory is not callable: {path}")
    return factory


def run(config: dict[str, Any]) -> dict[str, float]:
    """Build and train a model from a resolved configuration mapping.

    This function owns runtime initialization and shutdown, including cleanup
    when model construction or training fails.

    Args:
        config: Validated configuration, normally returned by
            :func:`generic_trainer.config.load_config`.

    Returns:
        Final training and validation metrics.

    Example:
        ``run(load_config("configs/smoke.yaml"))``
    """
    effective_environment = apply_environment(config["trainer"].get("environment"))
    diagnostics_cfg = config["trainer"].get("diagnostics", {})
    diagnostics_enabled = bool(diagnostics_cfg.get("enabled", False))
    runtime = initialize(
        timeout_seconds=int(diagnostics_cfg.get("collective_timeout_seconds", 1800)),
        diagnostics=diagnostics_enabled,
    )
    try:
        # Seed before model construction so initialization is reproducible.
        seed = int(config["run"]["seed"]) + runtime.rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model_cfg = {k: v for k, v in config["model"].items() if k != "factory"}
        data_cfg = {k: v for k, v in config["data"].items() if k != "factory"}
        model_factory = import_factory(config["model"]["factory"])
        if bool(diagnostics_cfg.get("fail_fast_capacity_check", False)):
            with torch.device("meta"):
                preview_model = model_factory(model_cfg)
            preview_size = model_size(preview_model)
            capacity = validate_capacity(
                int(preview_size["trainable_parameters"]), runtime.device.type, diagnostics_cfg.get("capacity", {})
            )
            print(json.dumps({"event": "capacity_preflight_passed", "rank": runtime.rank, **preview_size, **capacity}), flush=True)
            del preview_model
        model = model_factory(model_cfg)
        train_data, val_data = import_factory(config["data"]["factory"])(data_cfg, seed=int(config["run"]["seed"]))
        if runtime.is_main:
            count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            def dataset_length(dataset: Any) -> int | None:
                """Return a finite dataset length or ``None`` for a stream."""

                try:
                    return len(dataset)
                except TypeError:
                    return None

            print(json.dumps({"event": "startup", "device": str(runtime.device), "world_size": runtime.world_size,
                              "runtime_environment": effective_environment,
                              "parameters": count, "train_examples": dataset_length(train_data),
                              "validation_examples": dataset_length(val_data)}))
            output = Path(config["run"]["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
        return Trainer(model, train_data, val_data, config, runtime).fit()
    finally:
        shutdown(runtime)


def main() -> None:
    """Parse CLI arguments and execute one training run.

    Example:
        ``main()`` is invoked automatically by
        ``python -m generic_trainer.cli --config CONFIG``.
    """
    parser = argparse.ArgumentParser(description="Generic configuration-driven PyTorch trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
