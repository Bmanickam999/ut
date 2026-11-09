"""Model, configuration, capacity, and trainer-contract regression tests.

Run with ``python -m pytest tests/test_contracts.py -q``.
"""

from pathlib import Path

import torch
import yaml

from generic_trainer.config import load_config
from models.housing_llm.factory import build_datasets, build_model
from generic_trainer.telemetry import model_size
from generic_trainer.distributed import Runtime, reduce_mean
from generic_trainer.preflight import estimate_training_bytes, validate_capacity
from generic_trainer.optimizers import build_optimizer


ROOT = Path(__file__).parents[1]


def test_smoke_model_contract():
    cfg = load_config(ROOT / "configs/smoke.yaml")
    model = build_model({k: v for k, v in cfg["model"].items() if k != "factory"})
    train, validation = build_datasets({k: v for k, v in cfg["data"].items() if k != "factory"}, seed=7)
    batch = {key: value.unsqueeze(0) for key, value in train[0].items()}
    output = model(**batch)
    assert output["loss"].ndim == 0
    assert output["predictions"].shape == (1,)
    assert len(train) + len(validation) == 64


def test_500m_parameter_budget():
    cfg = load_config(ROOT / "configs/train_500m.yaml")
    # Validate the real module graph without allocating ~2 GB of fp32 weights.
    with torch.device("meta"):
        model = build_model({k: v for k, v in cfg["model"].items() if k != "factory"})
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 490_000_000 <= count <= 510_000_000
    report = model_size(model)
    assert report["trainable_parameters"] == count
    assert report["model_bytes"] == count * 4
    assert 1.8 < report["model_gib"] < 2.0


def test_1b_parameter_budget_without_allocating_weights():
    cfg = load_config(ROOT / "configs/train_1b.yaml")
    with torch.device("meta"):
        model = build_model({k: v for k, v in cfg["model"].items() if k != "factory"})
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 1_000_000_000 <= count <= 1_020_000_000


def test_capacity_preflight_fails_before_unsafe_allocation():
    config = {"bytes_per_parameter": 20.0, "activation_reserve_gib": 4.0, "max_memory_fraction": 0.9}
    required = estimate_training_bytes(1_000_000_000)
    assert required > 20_000_000_000
    try:
        validate_capacity(1_000_000_000, "cuda", config, available_bytes=16 * 1024**3)
    except RuntimeError as error:
        assert "capacity preflight failed" in str(error)
    else:
        raise AssertionError("unsafe 1B allocation was accepted")


def test_rejects_unknown_override():
    try:
        load_config(ROOT / "configs/smoke.yaml", ["trainer.typo=3"])
    except ValueError as error:
        assert "unknown override" in str(error)
    else:
        raise AssertionError("unknown configuration override was accepted")


def test_omitted_hyperparameters_resolve_to_documented_defaults(tmp_path: Path):
    """Minimal YAML receives explicit, checkpointable trainer defaults."""
    path = tmp_path / "minimal.yaml"
    path.write_text(yaml.safe_dump({
        "run": {"name": "minimal", "seed": 42, "output_dir": "artifacts/minimal"},
        "model": {"factory": "models.helloworld.new.factory:build_model"},
        "data": {"factory": "models.helloworld.new.factory:build_datasets", "num_examples": 64},
        "trainer": {},
    }))
    config = load_config(path)
    assert config["trainer"]["max_epochs"] == 1
    assert config["trainer"]["train_batch_size"] == 32
    assert config["trainer"]["optimizer"]["name"] == "adamw"
    assert config["trainer"]["optimizer"]["parameters"]["lr"] == 0.001
    assert config["trainer"]["scheduler"]["name"] == "cosine"
    assert config["data"]["shuffle_train"] is True


def test_reduce_mean_uses_portable_float32(monkeypatch):
    captured = {}
    original_tensor = torch.tensor

    def capture_tensor(*args, **kwargs):
        captured["dtype"] = kwargs.get("dtype")
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", capture_tensor)
    runtime = Runtime(torch.device("cpu"), rank=0, local_rank=0, world_size=1, distributed=False)
    assert reduce_mean(1.25, runtime) == 1.25
    assert captured["dtype"] == torch.float32


def test_yaml_selects_supported_core_optimizers():
    expected = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "nadam": torch.optim.NAdam,
        "radam": torch.optim.RAdam,
    }
    for name, optimizer_type in expected.items():
        model = torch.nn.Linear(2, 1)
        optimizer, resolved_name = build_optimizer(model, {"optimizer": {"name": name, "parameters": {"lr": 0.01}}})
        assert isinstance(optimizer, optimizer_type)
        assert resolved_name == name


def test_unknown_optimizer_and_arguments_fail():
    model = torch.nn.Linear(2, 1)
    for config, expected in [
        ({"name": "mystery", "parameters": {"lr": 0.1}}, "unsupported optimizer"),
        ({"name": "adam", "parameters": {"lr": 0.1, "typo": 3}}, "unsupported arguments"),
    ]:
        try:
            build_optimizer(model, {"optimizer": config})
        except (ValueError, RuntimeError) as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"invalid optimizer configuration was accepted: {config}")
