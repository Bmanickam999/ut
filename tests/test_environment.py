"""Runtime-environment validation and precedence tests."""

from __future__ import annotations

import pytest

from generic_trainer.environment import apply_environment


def test_environment_defaults_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """YAML defaults populate absent allowlisted environment variables."""

    monkeypatch.delenv("NCCL_DEBUG", raising=False)
    effective = apply_environment({"variables": {"NCCL_DEBUG": "INFO"}})
    assert effective == {"NCCL_DEBUG": "INFO"}


def test_existing_scheduler_value_wins_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cluster-injected values are preserved unless override is explicit."""

    monkeypatch.setenv("NCCL_DEBUG", "WARN")
    effective = apply_environment({
        "override_existing": False,
        "variables": {"NCCL_DEBUG": "INFO"},
    })
    assert effective["NCCL_DEBUG"] == "WARN"


def test_explicit_override_replaces_existing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-in override policy replaces an existing environment value."""

    monkeypatch.setenv("NCCL_DEBUG", "WARN")
    effective = apply_environment({
        "override_existing": True,
        "variables": {"NCCL_DEBUG": "INFO"},
    })
    assert effective["NCCL_DEBUG"] == "INFO"


def test_environment_rejects_unrelated_or_structured_values() -> None:
    """The YAML cannot mutate arbitrary process state or inject structures."""

    with pytest.raises(ValueError, match="not allowlisted"):
        apply_environment({"variables": {"PATH": "/tmp"}})
    with pytest.raises(ValueError, match="scalar"):
        apply_environment({"variables": {"NCCL_DEBUG": ["INFO"]}})
