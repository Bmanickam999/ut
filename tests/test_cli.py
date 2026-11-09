"""Behavioral tests for the installed ``llmdemo`` command interface.

Run with ``python -m pytest tests/test_cli.py -q``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmdemo_cli.main import main


def test_doctor_reports_runtime(capsys):
    assert main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["llmdemo"]
    assert report["pytorch"]
    assert isinstance(report["cuda_device_count"], int)
    assert report["gloo_available"] is True


def test_optimizers_lists_core_and_optional(capsys):
    assert main(["optimizers"]) == 0
    names = capsys.readouterr().out.splitlines()
    assert "adam" in names
    assert "adamw" in names
    assert "sgd" in names
    assert "distributed_shampoo (optional)" in names


def test_init_creates_runnable_yaml(tmp_path: Path):
    destination = tmp_path / "starter"
    assert main(["init", str(destination)]) == 0
    config = destination / "config.yaml"
    assert config.is_file()
    text = config.read_text()
    assert "models.helloworld.new.factory:build_model" in text
    assert "optimizer:" in text


def test_init_refuses_overwrite_without_force(tmp_path: Path):
    destination = tmp_path / "starter"
    main(["init", str(destination)])
    with pytest.raises(SystemExit, match="already exists"):
        main(["init", str(destination)])
    assert main(["init", str(destination), "--force"]) == 0


def test_multi_process_requires_cuda():
    with pytest.raises(SystemExit, match="requires --device cuda"):
        main(["train", "--config", "unused.yaml", "--device", "cpu", "--num-processes", "2"])


def test_process_count_must_be_positive():
    with pytest.raises(SystemExit, match="at least 1"):
        main(["train", "--config", "unused.yaml", "--num-processes", "0"])

