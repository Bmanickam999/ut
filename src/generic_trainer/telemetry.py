"""Model-size, rank-placement, and NVIDIA GPU utilization telemetry.

Example:
    ``model_size(model)`` returns exact parameter and buffer bytes, while
    ``HardwareTelemetry(runtime).placement_event(size)`` maps ranks to GPUs.
"""

from __future__ import annotations

import os
import socket
import subprocess
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from .distributed import Runtime


def model_size(model: nn.Module) -> dict[str, int | float]:
    """Return exact parameter and buffer size for the current dtype.

    Example:
        ``model_size(torch.nn.Linear(4, 2))["parameters"]`` returns ``10``.
    """
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    parameter_count = sum(value.numel() for value in parameters)
    trainable_count = sum(value.numel() for value in parameters if value.requires_grad)
    parameter_bytes = sum(value.numel() * value.element_size() for value in parameters)
    buffer_bytes = sum(value.numel() * value.element_size() for value in buffers)
    total_bytes = parameter_bytes + buffer_bytes
    return {
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "model_bytes": total_bytes,
        "model_gib": total_bytes / 1024**3,
    }


def _visible_physical_index(local_rank: int) -> str:
    """Map a logical CUDA rank through ``CUDA_VISIBLE_DEVICES``."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return str(local_rank)
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    return devices[local_rank] if local_rank < len(devices) else str(local_rank)


def query_nvidia_smi() -> list[dict[str, Any]]:
    """Read physical GPU telemetry, returning an empty list when unavailable."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    devices = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",", 5)]
        if len(values) != 6:
            continue
        index, uuid, name, utilization, used, total = values
        devices.append({
            "gpu_index": index,
            "uuid": uuid,
            "name": name,
            "utilization_percent": float(utilization),
            "memory_used_mib": float(used),
            "memory_total_mib": float(total),
        })
    return devices


@dataclass
class HardwareTelemetry:
    """Gather device placement and utilization from every process rank."""
    runtime: Runtime
    enabled: bool = True

    def placement_event(self, size: dict[str, Any]) -> dict[str, Any] | None:
        """Build a rank-to-device inventory event on rank zero."""
        placement = self._local_placement()
        placements = self._gather(placement)
        if not self.runtime.is_main:
            return None
        inventory = query_nvidia_smi() if self.enabled else []
        used = {str(item["physical_gpu_index"]) for item in placements if item["device_type"] == "cuda"}
        for gpu in inventory:
            ranks = [item["rank"] for item in placements if str(item["physical_gpu_index"]) == str(gpu["gpu_index"])]
            gpu["ranks"] = ranks
            gpu["utilized_by_trainer"] = bool(ranks)
        return {
            "event": "hardware_placement",
            **size,
            "world_size": self.runtime.world_size,
            "rank_placements": placements,
            "gpu_inventory": inventory,
            "trainer_gpu_count": len(used),
            "unused_gpu_indices": [gpu["gpu_index"] for gpu in inventory if not gpu["utilized_by_trainer"]],
            "estimated_total_model_gib_across_ranks": float(size["model_gib"]) * self.runtime.world_size,
        }

    def utilization_event(self, global_step: int) -> dict[str, Any] | None:
        """Build a point-in-time per-rank GPU utilization event on rank zero."""
        if not self.enabled:
            return None
        sample = self._local_sample()
        samples = self._gather(sample)
        if not self.runtime.is_main:
            return None
        inventory = query_nvidia_smi()
        rank_by_gpu = {str(item["physical_gpu_index"]): item["rank"] for item in samples if item["device_type"] == "cuda"}
        for gpu in inventory:
            gpu["rank"] = rank_by_gpu.get(str(gpu["gpu_index"]))
            gpu["utilized_by_trainer"] = gpu["rank"] is not None
        return {
            "event": "gpu_utilization",
            "global_step": global_step,
            "rank_samples": samples,
            "gpu_inventory": inventory,
        }

    def _local_placement(self) -> dict[str, Any]:
        """Describe this process's host, rank, device, and accelerator."""
        device = self.runtime.device
        result: dict[str, Any] = {
            "host": socket.gethostname(),
            "rank": self.runtime.rank,
            "local_rank": self.runtime.local_rank,
            "device": str(device),
            "device_type": device.type,
            "physical_gpu_index": _visible_physical_index(self.runtime.local_rank) if device.type == "cuda" else None,
        }
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            result.update(device_name=props.name, device_memory_gib=props.total_memory / 1024**3)
        return result

    def _local_sample(self) -> dict[str, Any]:
        """Add CUDA allocator statistics to local placement metadata."""
        sample = self._local_placement()
        if self.runtime.device.type == "cuda":
            sample.update(
                allocated_memory_mib=torch.cuda.memory_allocated(self.runtime.device) / 1024**2,
                reserved_memory_mib=torch.cuda.memory_reserved(self.runtime.device) / 1024**2,
                peak_allocated_memory_mib=torch.cuda.max_memory_allocated(self.runtime.device) / 1024**2,
            )
        return sample

    def _gather(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Gather Python telemetry dictionaries from all ranks."""
        if not self.runtime.distributed:
            return [value]
        gathered: list[dict[str, Any] | None] = [None] * self.runtime.world_size
        dist.all_gather_object(gathered, value)
        return [item for item in gathered if item is not None]
