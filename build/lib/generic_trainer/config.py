from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


REQUIRED = {
    "run": ["name", "seed", "output_dir"],
    "model": ["factory"],
    "data": ["factory", "num_examples"],
    "trainer": ["max_epochs", "train_batch_size", "precision"],
}


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    config = copy.deepcopy(config)
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
        raise ValueError("trainer requires either optimizer configuration or legacy learning_rate")
    return config
