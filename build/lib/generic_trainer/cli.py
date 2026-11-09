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
from .preflight import validate_capacity
from .telemetry import model_size


def import_factory(path: str) -> Callable[..., Any]:
    module_name, separator, attribute = path.partition(":")
    if not separator:
        raise ValueError(f"factory must use module:function syntax: {path}")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"factory is not callable: {path}")
    return factory


def run(config: dict[str, Any]) -> dict[str, float]:
    diagnostics_cfg = config["trainer"].get("diagnostics", {})
    diagnostics_enabled = bool(diagnostics_cfg.get("enabled", False))
    runtime = initialize(
        timeout_seconds=int(diagnostics_cfg.get("collective_timeout_seconds", 1800)),
        diagnostics=diagnostics_enabled,
    )
    try:
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
            print(json.dumps({"event": "startup", "device": str(runtime.device), "world_size": runtime.world_size,
                              "parameters": count, "train_examples": len(train_data), "validation_examples": len(val_data)}))
            output = Path(config["run"]["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
        return Trainer(model, train_data, val_data, config, runtime).fit()
    finally:
        shutdown(runtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic configuration-driven PyTorch trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
