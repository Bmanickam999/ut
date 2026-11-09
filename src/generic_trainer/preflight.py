"""Estimate training memory and reject predictably unsafe launches.

Example:
    ``validate_capacity(500_000_000, "cuda", policy)`` compares an
    approximate per-rank training footprint with available device memory.
"""

from __future__ import annotations

import os
from typing import Any

import torch


def available_memory_bytes(device_type: str, device_index: int = 0) -> int | None:
    """Return best-effort available bytes for a CUDA device or the host."""
    if device_type == "cuda":
        free, _ = torch.cuda.mem_get_info(device_index)
        return int(free)
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages
    except (AttributeError, OSError, ValueError):
        return None


def estimate_training_bytes(
    parameter_count: int,
    bytes_per_parameter: float = 20.0,
    activation_reserve_gib: float = 4.0,
) -> int:
    """Estimate per-rank bytes for weights, gradients, optimizer, and workspace.

    Example:
        ``estimate_training_bytes(1_000_000)`` applies the conservative
        default of 20 bytes per trainable parameter plus activation reserve.
    """
    return int(parameter_count * bytes_per_parameter + activation_reserve_gib * 1024**3)


def validate_capacity(
    parameter_count: int,
    device_type: str,
    config: dict[str, Any],
    available_bytes: int | None = None,
) -> dict[str, Any]:
    """Validate estimated memory against available capacity and policy.

    Returns a report for startup logs, or raises ``RuntimeError`` before model
    allocation when the estimate is unsafe or required memory data is absent.
    """
    required = estimate_training_bytes(
        parameter_count,
        float(config.get("bytes_per_parameter", 20.0)),
        float(config.get("activation_reserve_gib", 4.0)),
    )
    available = available_bytes if available_bytes is not None else available_memory_bytes(device_type)
    safety_fraction = float(config.get("max_memory_fraction", 0.9))
    report = {
        "parameter_count": parameter_count,
        "device_type": device_type,
        "estimated_required_bytes_per_rank": required,
        "estimated_required_gib_per_rank": required / 1024**3,
        "available_bytes": available,
        "available_gib": available / 1024**3 if available is not None else None,
        "max_memory_fraction": safety_fraction,
    }
    if available is None and bool(config.get("require_memory_check", True)):
        raise RuntimeError("capacity preflight could not determine available memory; refusing allocation")
    if available is not None and required > available * safety_fraction:
        raise RuntimeError(
            "capacity preflight failed: estimated training memory "
            f"{report['estimated_required_gib_per_rank']:.2f} GiB per rank exceeds "
            f"{safety_fraction:.0%} of available {report['available_gib']:.2f} GiB. "
            "Reduce model/batch size, use a larger accelerator, or adopt FSDP/ZeRO."
        )
    return report
