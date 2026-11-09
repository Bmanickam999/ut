"""Native and extensible distributed model-wrapping strategies.

Built-in strategies are ``single``, ``ddp``, and ``fsdp``. Tensor and pipeline
parallel layouts are model-specific, so they use ``strategy.factory`` to
return an object implementing the :class:`ModelStrategy` protocol below.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Protocol, runtime_checkable

from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .distributed import Runtime


@runtime_checkable
class ModelStrategy(Protocol):
    """Contract for model-specific tensor or pipeline partitioning."""

    def wrap(self, model: nn.Module, runtime: Runtime, config: dict[str, Any]) -> nn.Module:
        """Place or partition ``model`` for the supplied runtime."""


MODEL_PARALLEL_NAMES = {"none", "fsdp", "tensor_parallel", "pipeline_parallel"}


def resolve_parallelism(trainer_config: dict[str, Any], runtime: Runtime) -> dict[str, Any]:
    """Resolve user-facing model-parallel YAML into an execution strategy.

    ``model_parallelism.name=none`` intentionally resolves to DDP whenever
    ``world_size > 1`` and to a single process otherwise. Tensor/pipeline
    topology sizes must divide the launched world size.
    """

    model_parallel = dict(trainer_config.get("model_parallelism", {"name": "none"}))
    name = str(model_parallel.get("name", "none")).lower()
    if name not in MODEL_PARALLEL_NAMES:
        raise ValueError(f"unsupported trainer.model_parallelism.name: {name}")
    tp_size = int(model_parallel.get("tensor_parallel_size", 1))
    pp_size = int(model_parallel.get("pipeline_parallel_size", 1))
    if tp_size < 1 or pp_size < 1:
        raise ValueError("tensor_parallel_size and pipeline_parallel_size must be positive")
    if name == "none" and (tp_size != 1 or pp_size != 1):
        raise ValueError("model_parallelism.name=none requires tensor_parallel_size=1 and pipeline_parallel_size=1")
    if name == "fsdp" and (tp_size != 1 or pp_size != 1):
        raise ValueError("built-in FSDP cannot be combined with tensor/pipeline sizes; use a custom hybrid strategy")
    model_parallel_size = tp_size * pp_size
    if runtime.world_size % model_parallel_size:
        raise ValueError(
            f"world_size={runtime.world_size} must be divisible by model-parallel size {model_parallel_size}"
        )
    configured_dp = model_parallel.get("data_parallel_size", "auto")
    data_parallel_size = runtime.world_size // model_parallel_size
    if configured_dp != "auto" and int(configured_dp) != data_parallel_size:
        raise ValueError(
            f"data_parallel_size={configured_dp} conflicts with inferred size {data_parallel_size}"
        )
    resolved = {
        **model_parallel,
        "name": "ddp" if name == "none" and runtime.distributed else "single" if name == "none" else name,
        "model_parallel_name": name,
        "tensor_parallel_size": tp_size,
        "pipeline_parallel_size": pp_size,
        "data_parallel_size": data_parallel_size,
    }
    return resolved


def wrap_model(
    model: nn.Module,
    runtime: Runtime,
    strategy_config: dict[str, Any],
    import_factory: Callable[[str], Any] | None = None,
) -> tuple[nn.Module, str]:
    """Wrap a model with single-process, DDP, FSDP, or custom strategy.

    Custom strategies are the integration point for model-specific tensor and
    pipeline plans because layer partitioning cannot be inferred safely from
    an arbitrary ``nn.Module``.
    """

    name = str(strategy_config.get("name", "ddp" if runtime.distributed else "single")).lower()
    if name == "auto":
        name = "ddp" if runtime.distributed else "single"
    if name == "single":
        if runtime.distributed:
            raise ValueError("strategy=single cannot be used with world_size > 1")
        return model, name
    if name == "ddp":
        if not runtime.distributed:
            return model, "single"
        kwargs = {"device_ids": [runtime.local_rank], "output_device": runtime.local_rank} if runtime.device.type == "cuda" else {}
        return DistributedDataParallel(model, **kwargs), name
    if name == "fsdp":
        if not runtime.distributed:
            raise ValueError("strategy=fsdp requires world_size > 1")
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel
        except ImportError as error:
            raise RuntimeError("strategy=fsdp requires a PyTorch build with FSDP") from error
        from torch.distributed.fsdp import CPUOffload, ShardingStrategy

        options = strategy_config.get("parameters", {})
        sharding_name = str(options.get("sharding_strategy", "FULL_SHARD")).upper()
        if not hasattr(ShardingStrategy, sharding_name):
            raise ValueError(f"unsupported FSDP sharding_strategy: {sharding_name}")
        device_id = runtime.device if runtime.device.type == "cuda" else None
        wrapped = FullyShardedDataParallel(
            model,
            device_id=device_id,
            use_orig_params=bool(options.get("use_orig_params", True)),
            sharding_strategy=getattr(ShardingStrategy, sharding_name),
            cpu_offload=CPUOffload(offload_params=bool(options.get("cpu_offload", False))),
            limit_all_gathers=bool(options.get("limit_all_gathers", True)),
            forward_prefetch=bool(options.get("forward_prefetch", False)),
            sync_module_states=bool(options.get("sync_module_states", False)),
        )
        return wrapped, name
    if name in {"tensor_parallel", "pipeline_parallel", "custom"}:
        reference = strategy_config.get("factory")
        if not reference or import_factory is None:
            raise ValueError(f"strategy={name} requires trainer.strategy.factory=module:function")
        strategy = import_factory(str(reference))(strategy_config)
        if not isinstance(strategy, ModelStrategy):
            raise TypeError("strategy factory must return an object implementing wrap(model, runtime, config)")
        return strategy.wrap(model, runtime, strategy_config), name
    raise ValueError(f"unsupported strategy: {name}")


def full_state_dict_context(
    model: nn.Module, strategy_name: str, rank: int, *, rank0_only: bool = True
) -> ContextManager[Any]:
    """Return a context for gathering or loading an FSDP full state dict."""

    if strategy_name != "fsdp":
        return nullcontext()
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        StateDictType,
        FullyShardedDataParallel,
    )

    return FullyShardedDataParallel.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only),
    )


def optimizer_state_dict(model: nn.Module, optimizer: Any, strategy_name: str) -> dict[str, Any]:
    """Return a checkpointable optimizer state for standard or FSDP models."""

    if strategy_name != "fsdp":
        return optimizer.state_dict()
    from torch.distributed.fsdp import FullyShardedDataParallel

    return FullyShardedDataParallel.optim_state_dict(model, optimizer)


def load_optimizer_state_dict(
    model: nn.Module, optimizer: Any, state: dict[str, Any], strategy_name: str
) -> None:
    """Load standard state or transform full FSDP state for the local shard."""

    if strategy_name == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel

        state = FullyShardedDataParallel.optim_state_dict_to_load(model, optimizer, state)
    optimizer.load_state_dict(state)
