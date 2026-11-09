"""Validated runtime environment defaults applied before distributed startup.

Example::

    apply_environment({
        "override_existing": False,
        "variables": {"NCCL_DEBUG": "INFO"},
    })

Existing scheduler or shell values take precedence unless
``override_existing`` is explicitly enabled.
"""

from __future__ import annotations

import os
from typing import Any


ALLOWED_PREFIXES = ("NCCL_", "TORCH_", "CUDA_", "GLOO_", "OMP_", "FI_")


def apply_environment(config: dict[str, Any] | None) -> dict[str, str]:
    """Apply allowlisted environment defaults and return effective values.

    Args:
        config: Mapping containing ``variables`` and optional
            ``override_existing``. ``None`` performs no changes.

    Raises:
        ValueError: If a key is not a recognized training-runtime variable or
            if a value is structured rather than scalar.
    """

    if not config:
        return {}
    variables = config.get("variables", {})
    if not isinstance(variables, dict):
        raise ValueError("trainer.environment.variables must be a mapping")
    override = bool(config.get("override_existing", False))
    effective: dict[str, str] = {}
    for key, value in variables.items():
        name = str(key)
        if not name.startswith(ALLOWED_PREFIXES):
            raise ValueError(
                f"environment variable {name!r} is not allowlisted; expected prefixes {ALLOWED_PREFIXES}"
            )
        if isinstance(value, (dict, list, tuple)) or value is None:
            raise ValueError(f"environment variable {name!r} must have a scalar value")
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        if override or name not in os.environ:
            os.environ[name] = rendered
        effective[name] = os.environ[name]
    return effective
