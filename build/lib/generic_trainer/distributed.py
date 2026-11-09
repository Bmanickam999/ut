from __future__ import annotations

import os
from datetime import timedelta
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize(timeout_seconds: int = 1800, diagnostics: bool = False) -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    requested = os.environ.get("LLMDEMO_DEVICE", "auto").lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported LLMDEMO_DEVICE: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was explicitly requested but is not available")
    if requested == "mps" and distributed:
        raise RuntimeError("MPS does not support this trainer's multi-process DDP mode")

    use_cuda = requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
    use_mps = requested == "mps" or (
        requested == "auto"
        and not use_cuda
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and not distributed
    )
    if use_cuda:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    elif use_mps:
        device, backend = torch.device("mps"), "gloo"
    else:
        device, backend = torch.device("cpu"), "gloo"
    if distributed and not dist.is_initialized():
        if diagnostics:
            print(f'DIAGNOSTIC {{"event":"process_group_init_start","rank":{rank},"backend":"{backend}"}}', flush=True)
        dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(seconds=timeout_seconds))
        if diagnostics:
            print(f'DIAGNOSTIC {{"event":"process_group_init_complete","rank":{rank},"backend":"{backend}"}}', flush=True)
    return Runtime(device, rank, local_rank, world_size, distributed)


def reduce_mean(value: float, runtime: Runtime) -> float:
    # MPS does not support float64. Metric aggregation does not need double
    # precision, and float32 is portable across MPS, CUDA/NCCL, and CPU/Gloo.
    tensor = torch.tensor(value, dtype=torch.float32, device=runtime.device)
    if runtime.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= runtime.world_size
    return float(tensor.item())


def shutdown(runtime: Runtime) -> None:
    if runtime.distributed and dist.is_initialized():
        dist.destroy_process_group()
