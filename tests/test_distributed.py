"""CPU/GPU DDP and collective-communication integration tests.

Run CPU coverage with ``python -m pytest tests/test_distributed.py -q``.
CUDA/NCCL tests activate automatically when two CUDA devices are available.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.multiprocessing as mp

from tests.ddp_workers import collective_worker, fsdp_worker, gradient_sync_worker, trainer_worker


WORLD_SIZE = 2


def free_local_port() -> int:
    """Reserve an ephemeral localhost port for one short-lived process group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def spawn(worker: Callable[..., None], output_dir: Path, *, force_cpu: bool = True) -> list[dict]:
    mp.spawn(
        worker,
        args=(WORLD_SIZE, free_local_port(), str(output_dir), force_cpu),
        nprocs=WORLD_SIZE,
        join=True,
    )
    return [json.loads((output_dir / f"rank-{rank}.json").read_text()) for rank in range(WORLD_SIZE)]


@pytest.mark.distributed
def test_cpu_gloo_collective_communications(tmp_path: Path) -> None:
    results = spawn(collective_worker, tmp_path / "collectives")
    assert results == [{"backend": "gloo", "mean": 1.5}] * WORLD_SIZE


@pytest.mark.distributed
def test_cpu_ddp_synchronizes_gradients(tmp_path: Path) -> None:
    results = spawn(gradient_sync_worker, tmp_path / "gradients")
    assert results == [{"gradient": 5.0, "device": "cpu"}] * WORLD_SIZE


@pytest.mark.distributed
def test_generic_trainer_end_to_end_under_cpu_ddp(tmp_path: Path) -> None:
    output = tmp_path / "trainer"
    mp.spawn(
        trainer_worker,
        args=(WORLD_SIZE, free_local_port(), str(output)),
        nprocs=WORLD_SIZE,
        join=True,
    )
    assert (output / "best.pt").is_file()
    assert (output / "last.pt").is_file()
    assert (output / "metrics.jsonl").is_file()
    assert len(list(output.glob("rank-*.json"))) == WORLD_SIZE


@pytest.mark.distributed
@pytest.mark.gpu
def test_gpu_nccl_collective_communications(tmp_path: Path) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip("requires at least two CUDA GPUs")
    results = spawn(collective_worker, tmp_path / "nccl-collectives", force_cpu=False)
    assert results == [{"backend": "nccl", "mean": 1.5}] * WORLD_SIZE


@pytest.mark.distributed
@pytest.mark.gpu
def test_gpu_ddp_synchronizes_gradients(tmp_path: Path) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip("requires at least two CUDA GPUs")
    results = spawn(gradient_sync_worker, tmp_path / "nccl-gradients", force_cpu=False)
    assert results == [{"gradient": 5.0, "device": "cuda"}] * WORLD_SIZE


@pytest.mark.distributed
@pytest.mark.gpu
def test_gpu_fsdp_forward_backward(tmp_path: Path) -> None:
    """FSDP performs a real sharded CUDA update when two GPUs are available."""

    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip("requires at least two CUDA GPUs")
    results = spawn(fsdp_worker, tmp_path / "fsdp", force_cpu=False)
    assert results == [{"strategy": "fsdp", "device": "cuda"}] * WORLD_SIZE
