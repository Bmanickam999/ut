from __future__ import annotations

import inspect
from typing import Any, Iterable

import torch
from torch import nn


CORE_OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "nadam": torch.optim.NAdam,
    "radam": torch.optim.RAdam,
    "rmsprop": torch.optim.RMSprop,
    "sgd": torch.optim.SGD,
}


def _validate_arguments(factory: type, arguments: dict[str, Any]) -> None:
    signature = inspect.signature(factory.__init__)
    accepts_extra = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    allowed = {name for name in signature.parameters if name not in {"self", "params"}}
    unknown = set(arguments) - allowed
    if unknown and not accepts_extra:
        raise ValueError(f"unsupported arguments for {factory.__name__}: {sorted(unknown)}; allowed: {sorted(allowed)}")


def _normalize(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    if "betas" in normalized:
        normalized["betas"] = tuple(float(value) for value in normalized["betas"])
    return normalized


def build_optimizer(model: nn.Module, trainer_config: dict[str, Any]) -> tuple[torch.optim.Optimizer, str]:
    """Build a validated optimizer from YAML, with backward-compatible defaults."""
    config = trainer_config.get("optimizer")
    if config is None:
        config = {
            "name": "adamw",
            "parameters": {
                "lr": float(trainer_config["learning_rate"]),
                "weight_decay": float(trainer_config.get("weight_decay", 0.0)),
            },
        }
    if not isinstance(config, dict) or not isinstance(config.get("name"), str):
        raise ValueError("trainer.optimizer must be a mapping containing a string 'name'")
    name = config["name"].lower().replace("-", "_")
    name = {"adam_w": "adamw", "r_adam": "radam", "n_adam": "nadam"}.get(name, name)
    arguments = _normalize(config.get("parameters", {}))
    if "lr" not in arguments:
        raise ValueError(f"trainer.optimizer.parameters.lr is required for {name}")

    if name in CORE_OPTIMIZERS:
        factory = CORE_OPTIMIZERS[name]
        _validate_arguments(factory, arguments)
        return factory(model.parameters(), **arguments), name
    if name in {"shampoo", "distributed_shampoo"}:
        return _build_shampoo(model.parameters(), arguments, config), "distributed_shampoo"
    raise ValueError(f"unsupported optimizer '{name}'; supported: {sorted(CORE_OPTIMIZERS)} plus distributed_shampoo")


def _build_shampoo(
    parameters: Iterable[nn.Parameter], arguments: dict[str, Any], config: dict[str, Any]
) -> torch.optim.Optimizer:
    try:
        from distributed_shampoo import AdamPreconditionerConfig, DistributedShampoo
    except ImportError as error:
        raise RuntimeError(
            "optimizer 'distributed_shampoo' requires Meta's optional distributed_shampoo package. "
            "Install a version compatible with this environment from "
            "https://github.com/facebookresearch/optimizers and rerun; no fallback optimizer was used."
        ) from error

    grafting = config.get("grafting")
    if grafting:
        grafting_name = str(grafting.get("name", "adam")).lower()
        if grafting_name != "adam":
            raise ValueError("the current Shampoo adapter supports grafting.name=adam only")
        arguments["grafting_config"] = AdamPreconditionerConfig(
            beta2=float(grafting.get("beta2", 0.999)),
            epsilon=float(grafting.get("epsilon", 1e-8)),
        )
    _validate_arguments(DistributedShampoo, arguments)
    return DistributedShampoo(parameters, **arguments)
