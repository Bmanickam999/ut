"""Load, override, and validate Generic Trainer YAML configuration.

Example:
    ``load_config("configs/smoke.yaml", ["trainer.max_epochs=2"])``
    returns a deep-copied mapping with the validated override applied.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TRAINER: dict[str, Any] = {
    "max_epochs": 1,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "scheduler": {"name": "cosine", "warmup_steps": 0, "minimum_lr_ratio": 0.1},
    "max_grad_norm": 1.0,
    "precision": "auto",
    "compile": False,
    "num_workers": 0,
    "log_every_steps": 20,
    "telemetry": {"enabled": True, "log_every_steps": 20},
    "save_every_epochs": 1,
    "resume_from": None,
    "early_stopping": {"metric": "val_loss", "mode": "min", "patience": 3, "min_delta": 0.0},
    "safety": {"stop_on_non_finite_loss": True, "max_loss": 1_000_000.0, "max_consecutive_skips": 1},
    "model_parallelism": {
        "name": "none",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": "auto",
    },
}

DEFAULT_OPTIMIZER: dict[str, Any] = {
    "name": "adamw",
    "parameters": {"lr": 0.001, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1.0e-8},
}

DEFAULT_DATA_POLICY: dict[str, Any] = {
    "shuffle_train": True,
    "shuffle_validation": False,
    "drop_last": False,
}


def _apply_defaults(target: dict[str, Any], defaults: dict[str, Any]) -> None:
    """Recursively copy missing default values without replacing user input."""
    for key, value in defaults.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(target[key], dict):
            _apply_defaults(target[key], value)


REQUIRED = {
    "run": ["name", "seed", "output_dir"],
    "model": ["factory"],
    "data": ["factory", "num_examples"],
    "trainer": [],
}


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Read a YAML file and apply type-aware dotted-key overrides.

    Args:
        path: YAML configuration path.
        overrides: Optional values such as ``trainer.max_epochs=3``. Values
            are decoded with YAML rules, so numbers and booleans retain type.

    Returns:
        A validated, mutable configuration dictionary.

    Raises:
        ValueError: If YAML shape, override paths, or required fields are
            invalid.

    Example:
        ``cfg = load_config("configs/smoke.yaml", ["data.num_examples=64"])``
    """
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    config = copy.deepcopy(config)
    config.setdefault("trainer", {})
    config.setdefault("data", {})
    scheduler_was_explicit = "scheduler" in config["trainer"]
    _apply_defaults(config["trainer"], DEFAULT_TRAINER)
    if not scheduler_was_explicit and "warmup_steps" in config["trainer"]:
        config["trainer"]["scheduler"]["warmup_steps"] = int(config["trainer"]["warmup_steps"])
    _apply_defaults(config["data"], DEFAULT_DATA_POLICY)
    if "optimizer" not in config["trainer"] and "learning_rate" not in config["trainer"]:
        config["trainer"]["optimizer"] = copy.deepcopy(DEFAULT_OPTIMIZER)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value: {item}")
        dotted, raw = item.split("=", 1)
        cursor = config
        parts = dotted.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise ValueError(f"unknown override path: {dotted}")
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise ValueError(f"unknown override key: {dotted}")
        cursor[parts[-1]] = yaml.safe_load(raw)
    for section, keys in REQUIRED.items():
        if section not in config:
            raise ValueError(f"missing configuration section: {section}")
        missing = [key for key in keys if key not in config[section]]
        if missing:
            raise ValueError(f"{section} is missing keys: {missing}")
    trainer = config["trainer"]
    if "optimizer" not in trainer and "learning_rate" not in trainer:
        raise ValueError("trainer requires optimizer configuration or legacy learning_rate")
    scheduler = trainer.get("scheduler", {"name": "cosine"})
    if not isinstance(scheduler, dict):
        raise ValueError("trainer.scheduler must be a mapping")
    scheduler_name = str(scheduler.get("name", "cosine")).lower()
    if scheduler_name not in {"constant", "cosine", "linear"}:
        raise ValueError(f"unsupported trainer.scheduler.name: {scheduler_name}")
    minimum_lr_ratio = float(scheduler.get("minimum_lr_ratio", 0.1))
    if not 0.0 <= minimum_lr_ratio <= 1.0:
        raise ValueError("trainer.scheduler.minimum_lr_ratio must be between 0 and 1")
    strategy = trainer.get("strategy", {})
    strategy_name = str(strategy.get("name", "auto")).lower()
    supported = {"auto", "single", "ddp", "fsdp", "tensor_parallel", "pipeline_parallel", "custom"}
    if strategy_name not in supported:
        raise ValueError(f"unsupported trainer.strategy.name: {strategy_name}")
    if strategy_name in {"tensor_parallel", "pipeline_parallel", "custom"} and not strategy.get("factory"):
        raise ValueError(f"trainer.strategy.factory is required for {strategy_name}")
    model_parallel = trainer.get("model_parallelism", {"name": "none"})
    if not isinstance(model_parallel, dict):
        raise ValueError("trainer.model_parallelism must be a mapping")
    model_parallel_name = str(model_parallel.get("name", "none")).lower()
    model_parallel_names = {"none", "fsdp", "tensor_parallel", "pipeline_parallel"}
    if model_parallel_name not in model_parallel_names:
        raise ValueError(f"unsupported trainer.model_parallelism.name: {model_parallel_name}")
    if model_parallel_name in {"tensor_parallel", "pipeline_parallel"} and not model_parallel.get("factory"):
        raise ValueError(f"trainer.model_parallelism.factory is required for {model_parallel_name}")
    for size_name in ("tensor_parallel_size", "pipeline_parallel_size"):
        if int(model_parallel.get(size_name, 1)) < 1:
            raise ValueError(f"trainer.model_parallelism.{size_name} must be positive")
    if model_parallel_name == "none" and any(
        int(model_parallel.get(size_name, 1)) != 1
        for size_name in ("tensor_parallel_size", "pipeline_parallel_size")
    ):
        raise ValueError("model_parallelism.name=none requires tensor/pipeline sizes of 1")
    if "model_parallelism" in trainer and strategy and strategy_name not in {"auto", "ddp"}:
        raise ValueError("use trainer.model_parallelism instead of combining it with legacy trainer.strategy")
    environment = trainer.get("environment", {})
    if environment and not isinstance(environment.get("variables", {}), dict):
        raise ValueError("trainer.environment.variables must be a mapping")
    data = config["data"]
    for field in ("shuffle_train", "shuffle_validation", "drop_last"):
        if not isinstance(data[field], bool):
            raise ValueError(f"data.{field} must be a boolean")
    return config
