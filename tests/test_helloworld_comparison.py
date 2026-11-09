"""Contracts shared by the coupled and decoupled hello-world demonstrations."""

from __future__ import annotations

import torch
import pytest

from models.helloworld.before.helloworld import ToyDataset as BeforeDataset
from models.helloworld.new.data import ToyDataset as AfterDataset
from models.helloworld.compare import _difference
from models.simple_helloworld.before.helloworld import ToyDataset as SimpleBeforeDataset
from models.simple_helloworld.new.data import ToyDataset as SimpleAfterDataset
from generic_trainer.engine import Trainer


def test_before_and_after_receive_identical_examples() -> None:
    """Prove the visual comparison uses the same deterministic data contract."""
    before = BeforeDataset(32, seed=42)
    after = AfterDataset(32, seed=42)
    assert torch.equal(before.inputs, after.inputs)
    assert torch.equal(before.labels, after.labels)


def test_dataset_generation_is_independent_of_requested_size() -> None:
    """The first examples and target function remain fixed at leadership scale."""
    small = AfterDataset(16, seed=42)
    large = AfterDataset(64, seed=42)
    assert torch.equal(small.inputs, large.inputs[:16])
    assert torch.equal(small.labels, large.labels[:16])


def test_constant_scheduler_preserves_configured_learning_rate() -> None:
    """A constant YAML scheduler must return the base learning-rate ratio."""
    trainer = Trainer.__new__(Trainer)
    trainer.tcfg = {"scheduler": {"name": "constant", "warmup_steps": 0}}
    assert [trainer._schedule(step, 100) for step in (0, 50, 99)] == [1.0, 1.0, 1.0]


def test_cosine_scheduler_remains_available() -> None:
    """Central policy can still select cosine decay without model changes."""
    trainer = Trainer.__new__(Trainer)
    trainer.tcfg = {"scheduler": {"name": "cosine", "warmup_steps": 0, "minimum_lr_ratio": 0.1}}
    assert trainer._schedule(0, 100) == 1.0
    assert trainer._schedule(100, 100) == 0.1


def test_loss_difference_uses_before_as_percentage_baseline() -> None:
    """Leadership comparison reports both absolute and signed percentages."""
    result = _difference(0.5, 0.4)
    assert result["absolute_difference"] == pytest.approx(0.1)
    assert result["percentage_difference"] == pytest.approx(-20.0)
    assert result["absolute_percentage_difference"] == pytest.approx(20.0)
    assert result["Loss Improved"] == "Yes"
    assert _difference(0.4, 0.5)["Loss Improved"] == "No"


def test_simple_helloworld_preserves_small_shared_dataset_contract() -> None:
    """The preserved 128-example demo supplies identical records to both paths."""
    before = SimpleBeforeDataset(128, seed=42)
    after = SimpleAfterDataset(128, seed=42)
    assert len(before) == len(after) == 128
    assert torch.equal(before.inputs, after.inputs)
    assert torch.equal(before.labels, after.labels)
