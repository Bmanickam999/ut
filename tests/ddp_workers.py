"""Top-level multiprocessing workers used by the DDP integration tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset

from generic_trainer.distributed import initialize, reduce_mean, shutdown
from generic_trainer.engine import Trainer
from generic_trainer.strategies import wrap_model


def configure_process(rank: int, world_size: int, port: int, force_cpu: bool) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    if force_cpu:
        # initialize() sees a CPU-only process even on a GPU development host.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def collective_worker(rank: int, world_size: int, port: int, output_dir: str, force_cpu: bool) -> None:
    configure_process(rank, world_size, port, force_cpu)
    runtime = initialize()
    try:
        expected_backend = "gloo" if force_cpu else "nccl"
        assert dist.get_backend() == expected_backend

        # SUM all-reduce: 1 + ... + world_size on every rank.
        reduced = torch.tensor(float(rank + 1), device=runtime.device)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        expected_sum = world_size * (world_size + 1) / 2
        assert reduced.item() == expected_sum

        # Broadcast rank-zero state and gather per-rank values.
        broadcast = torch.tensor(123.0 if rank == 0 else -1.0, device=runtime.device)
        dist.broadcast(broadcast, src=0)
        assert broadcast.item() == 123.0
        gathered = [torch.zeros(1, dtype=torch.int64, device=runtime.device) for _ in range(world_size)]
        dist.all_gather(gathered, torch.tensor([rank], dtype=torch.int64, device=runtime.device))
        assert [int(value.item()) for value in gathered] == list(range(world_size))

        # Exercise the trainer's collective helper as well as raw torch calls.
        mean = reduce_mean(float(rank + 1), runtime)
        assert mean == (world_size + 1) / 2
        dist.barrier()
        _write_result(output_dir, rank, {"backend": expected_backend, "mean": mean})
    finally:
        shutdown(runtime)


def gradient_sync_worker(rank: int, world_size: int, port: int, output_dir: str, force_cpu: bool) -> None:
    configure_process(rank, world_size, port, force_cpu)
    runtime = initialize()
    try:
        model = nn.Linear(1, 1, bias=False).to(runtime.device)
        with torch.no_grad():
            model.weight.fill_(1.0)
        kwargs = {"device_ids": [rank], "output_device": rank} if runtime.device.type == "cuda" else {}
        ddp = DistributedDataParallel(model, **kwargs)

        # Rank-specific inputs produce local gradients 2 and 8. DDP must average
        # those to 5 on both ranks before the optimizer step.
        value = torch.tensor([[float(rank + 1)]], device=runtime.device)
        ddp(value).pow(2).sum().backward()
        gradient = float(model.weight.grad.item())
        assert gradient == 5.0
        dist.barrier()
        _write_result(output_dir, rank, {"gradient": gradient, "device": runtime.device.type})
    finally:
        shutdown(runtime)


class TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, size: int = 8) -> None:
        self.values = torch.arange(1, size + 1, dtype=torch.float32).unsqueeze(1) / size

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = self.values[index]
        return {"features": value, "targets": 2 * value}


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        predictions = self.linear(features)
        return {"loss": torch.nn.functional.mse_loss(predictions, targets), "predictions": predictions}


def trainer_worker(rank: int, world_size: int, port: int, output_dir: str) -> None:
    configure_process(rank, world_size, port, force_cpu=True)
    runtime = initialize()
    config: dict[str, Any] = {
        "run": {"name": "ddp-test", "seed": 17, "output_dir": output_dir},
        "trainer": {
            "max_epochs": 1,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "max_grad_norm": 1.0,
            "precision": "fp32",
            "compile": False,
            "num_workers": 0,
            "log_every_steps": 1,
            "save_every_epochs": 1,
            "resume_from": None,
            "early_stopping": {"metric": "val_loss", "mode": "min", "patience": 2, "min_delta": 0.0},
            "safety": {"stop_on_non_finite_loss": True, "max_loss": 100.0, "max_consecutive_skips": 1},
        },
    }
    try:
        summary = Trainer(TinyModel(), TinyDataset(8), TinyDataset(4), config, runtime).fit()
        assert torch.isfinite(torch.tensor(summary["val_loss"]))
        dist.barrier()
        _write_result(output_dir, rank, summary)
    finally:
        shutdown(runtime)


def fsdp_worker(rank: int, world_size: int, port: int, output_dir: str, force_cpu: bool) -> None:
    """Exercise a native FSDP forward/backward update on CUDA."""

    configure_process(rank, world_size, port, force_cpu)
    runtime = initialize()
    try:
        model, strategy = wrap_model(nn.Linear(4, 2).to(runtime.device), runtime, {"name": "fsdp"})
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        model(torch.ones(2, 4, device=runtime.device)).sum().backward()
        optimizer.step()
        dist.barrier()
        _write_result(output_dir, rank, {"strategy": strategy, "device": runtime.device.type})
    finally:
        shutdown(runtime)


def _write_result(output_dir: str, rank: int, result: dict[str, Any]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
