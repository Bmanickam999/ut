"""Tests for selectable model-parallel strategies and topology validation."""

from __future__ import annotations

import pytest
import torch

from generic_trainer.distributed import Runtime
from generic_trainer.strategies import resolve_parallelism, wrap_model
from models.helloworld.new.helloworld import ToyModel


def test_custom_parallel_strategy_contract() -> None:
    """Tensor and pipeline plans can wrap model-specific module graphs."""

    class Strategy:
        def wrap(self, model, runtime, config):
            model.strategy_marker = config["name"]
            return model

    model, name = wrap_model(
        ToyModel(),
        Runtime(torch.device("cpu"), 0, 0, 1, False),
        {"name": "tensor_parallel", "factory": "demo:factory"},
        lambda reference: lambda config: Strategy(),
    )
    assert name == "tensor_parallel"
    assert model.strategy_marker == "tensor_parallel"


@pytest.mark.parametrize("name", ["tensor_parallel", "pipeline_parallel", "custom"])
def test_model_specific_strategies_require_factory(name: str) -> None:
    """Partitioning strategies never guess unsafe model split points."""

    with pytest.raises(ValueError, match="strategy.factory"):
        wrap_model(ToyModel(), Runtime(torch.device("cpu"), 0, 0, 1, False), {"name": name})


def test_fsdp_requires_a_distributed_process_group() -> None:
    """FSDP cannot silently degrade to a single unsharded process."""

    with pytest.raises(ValueError, match="world_size"):
        wrap_model(ToyModel(), Runtime(torch.device("cpu"), 0, 0, 1, False), {"name": "fsdp"})


def test_none_model_parallelism_resolves_to_ddp() -> None:
    """The default disables model parallelism but retains data parallelism."""

    resolved = resolve_parallelism(
        {"model_parallelism": {"name": "none"}},
        Runtime(torch.device("cpu"), 0, 0, 8, True),
    )
    assert resolved["name"] == "ddp"
    assert resolved["model_parallel_name"] == "none"
    assert resolved["data_parallel_size"] == 8
    assert resolved["tensor_parallel_size"] == 1
    assert resolved["pipeline_parallel_size"] == 1


def test_parallel_topology_infers_data_parallel_size() -> None:
    """Tensor/pipeline dimensions reserve remaining ranks for data parallelism."""

    resolved = resolve_parallelism(
        {"model_parallelism": {
            "name": "tensor_parallel",
            "factory": "demo:strategy",
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "data_parallel_size": "auto",
        }},
        Runtime(torch.device("cuda", 0), 0, 0, 8, True),
    )
    assert resolved["data_parallel_size"] == 2


def test_invalid_parallel_topology_fails_before_model_wrapping() -> None:
    """A world size that cannot form the requested topology is rejected."""

    with pytest.raises(ValueError, match="must be divisible"):
        resolve_parallelism(
            {"model_parallelism": {
                "name": "pipeline_parallel", "factory": "demo:strategy",
                "pipeline_parallel_size": 3,
            }},
            Runtime(torch.device("cuda", 0), 0, 0, 8, True),
        )
