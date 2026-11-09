"""Optimizer registry, validation, update, and state round-trip tests.

Run with ``python -m pytest tests/test_optimizers.py -q``.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from generic_trainer.optimizers import CORE_OPTIMIZERS, build_optimizer


@pytest.mark.parametrize(
    ("name", "optimizer_type", "parameters"),
    [
        ("sgd", torch.optim.SGD, {"lr": 0.1, "momentum": 0.9, "nesterov": True}),
        ("adam", torch.optim.Adam, {"lr": 0.001, "betas": [0.8, 0.95], "eps": 1e-7}),
        ("adamw", torch.optim.AdamW, {"lr": 0.001, "weight_decay": 0.1}),
        ("rmsprop", torch.optim.RMSprop, {"lr": 0.01, "alpha": 0.95, "momentum": 0.1}),
        ("adagrad", torch.optim.Adagrad, {"lr": 0.01, "lr_decay": 0.01}),
        ("adadelta", torch.optim.Adadelta, {"lr": 1.0, "rho": 0.8}),
        ("nadam", torch.optim.NAdam, {"lr": 0.002, "momentum_decay": 0.004}),
        ("radam", torch.optim.RAdam, {"lr": 0.002, "decoupled_weight_decay": True}),
    ],
)
def test_builds_each_core_optimizer(name, optimizer_type, parameters):
    model = torch.nn.Linear(3, 1)
    optimizer, resolved = build_optimizer(model, {"optimizer": {"name": name, "parameters": parameters}})
    assert isinstance(optimizer, optimizer_type)
    assert resolved == name
    assert optimizer.param_groups[0]["lr"] == parameters["lr"]


def test_optimizer_names_are_case_and_hyphen_tolerant():
    model = torch.nn.Linear(2, 1)
    optimizer, name = build_optimizer(
        model, {"optimizer": {"name": "ADAM-W", "parameters": {"lr": 0.01}}}
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert name == "adamw"


def test_yaml_beta_list_is_normalized_to_tuple():
    optimizer, _ = build_optimizer(
        torch.nn.Linear(2, 1),
        {"optimizer": {"name": "adam", "parameters": {"lr": 0.01, "betas": [0.7, 0.8]}}},
    )
    assert optimizer.param_groups[0]["betas"] == (0.7, 0.8)


def test_legacy_learning_rate_builds_adamw():
    optimizer, name = build_optimizer(
        torch.nn.Linear(2, 1), {"learning_rate": 0.003, "weight_decay": 0.2}
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert name == "adamw"
    assert optimizer.param_groups[0]["lr"] == 0.003
    assert optimizer.param_groups[0]["weight_decay"] == 0.2


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"optimizer": "adam"}, "must be a mapping"),
        ({"optimizer": {}}, "must be a mapping"),
        ({"optimizer": {"name": "adam", "parameters": {}}}, "parameters.lr is required"),
        ({"optimizer": {"name": "unknown", "parameters": {"lr": 0.1}}}, "unsupported optimizer"),
        ({"optimizer": {"name": "adam", "parameters": {"lr": 0.1, "lern_rate": 2}}}, "unsupported arguments"),
    ],
)
def test_invalid_optimizer_configuration_fails(config, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        build_optimizer(torch.nn.Linear(2, 1), config)


@pytest.mark.parametrize("name", sorted(CORE_OPTIMIZERS))
def test_optimizer_performs_a_real_parameter_update(name):
    torch.manual_seed(4)
    model = torch.nn.Linear(2, 1)
    before = model.weight.detach().clone()
    optimizer, _ = build_optimizer(model, {"optimizer": {"name": name, "parameters": {"lr": 0.01}}})
    loss = model(torch.tensor([[1.0, -2.0]])).pow(2).sum()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.weight.detach())


def test_optimizer_state_dict_round_trip():
    model = torch.nn.Linear(2, 1)
    config = {"optimizer": {"name": "adamw", "parameters": {"lr": 0.01}}}
    optimizer, _ = build_optimizer(model, config)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    state = optimizer.state_dict()

    restored, _ = build_optimizer(model, config)
    restored.load_state_dict(state)
    assert restored.state_dict()["state"].keys() == state["state"].keys()
    assert restored.param_groups[0]["lr"] == 0.01


def test_shampoo_missing_dependency_fails_without_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "distributed_shampoo", None)
    with pytest.raises(RuntimeError, match="requires Meta's optional"):
        build_optimizer(
            torch.nn.Linear(2, 1),
            {"optimizer": {"name": "distributed_shampoo", "parameters": {"lr": 0.001}}},
        )


def test_shampoo_adapter_propagates_parameters_and_grafting(monkeypatch):
    class FakeGrafting:
        def __init__(self, beta2: float, epsilon: float):
            self.beta2, self.epsilon = beta2, epsilon

    class FakeShampoo(torch.optim.Optimizer):
        def __init__(
            self,
            params,
            lr: float,
            betas=(0.9, 0.999),
            epsilon: float = 1e-8,
            weight_decay: float = 0.0,
            grafting_config=None,
        ):
            self.received = {
                "lr": lr,
                "betas": betas,
                "epsilon": epsilon,
                "weight_decay": weight_decay,
                "grafting_config": grafting_config,
            }
            super().__init__(params, {"lr": lr})

        def step(self, closure=None):
            return None

    fake_module = types.ModuleType("distributed_shampoo")
    fake_module.AdamPreconditionerConfig = FakeGrafting
    fake_module.DistributedShampoo = FakeShampoo
    monkeypatch.setitem(sys.modules, "distributed_shampoo", fake_module)

    optimizer, name = build_optimizer(
        torch.nn.Linear(2, 1),
        {
            "optimizer": {
                "name": "shampoo",
                "parameters": {
                    "lr": 0.002,
                    "betas": [0.8, 0.9],
                    "epsilon": 1e-10,
                    "weight_decay": 0.01,
                },
                "grafting": {"name": "adam", "beta2": 0.97, "epsilon": 1e-9},
            }
        },
    )
    assert name == "distributed_shampoo"
    assert optimizer.received["betas"] == (0.8, 0.9)
    assert optimizer.received["grafting_config"].beta2 == 0.97
    assert optimizer.received["grafting_config"].epsilon == 1e-9


def test_shampoo_rejects_unknown_grafting_strategy(monkeypatch):
    fake_module = types.ModuleType("distributed_shampoo")
    fake_module.AdamPreconditionerConfig = object
    fake_module.DistributedShampoo = object
    monkeypatch.setitem(sys.modules, "distributed_shampoo", fake_module)
    with pytest.raises(ValueError, match="grafting.name=adam only"):
        build_optimizer(
            torch.nn.Linear(2, 1),
            {
                "optimizer": {
                    "name": "shampoo",
                    "parameters": {"lr": 0.001},
                    "grafting": {"name": "sgd"},
                }
            },
        )
