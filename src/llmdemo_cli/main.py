"""Installed ``llmdemo`` command-line interface.

Examples:
    ``llmdemo doctor`` inspects the local runtime, and
    ``llmdemo train --config config.yaml --device cpu`` starts training.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import Sequence

import torch

from generic_trainer.cli import run
from generic_trainer.config import load_config
from generic_trainer.optimizers import CORE_OPTIMIZERS
from llmdemo_cli import __version__


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level parser and all supported subcommands."""
    parser = argparse.ArgumentParser(prog="llmdemo", description="Portable generic PyTorch trainer CLI")
    parser.add_argument("--version", action="version", version=f"llmdemo {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="Verify Python, PyTorch, devices, and distributed backends")
    commands.add_parser("optimizers", help="List YAML-selectable optimizers")

    init = commands.add_parser("init", help="Create a runnable starter configuration")
    init.add_argument("directory", nargs="?", default="llmdemo-run")
    init.add_argument("--force", action="store_true", help="Overwrite an existing starter config")

    train = commands.add_parser("train", help="Train from a YAML configuration")
    train.add_argument("--config", required=True)
    train.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    train.add_argument("--num-processes", type=int, default=1, metavar="N")
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    compare = commands.add_parser("compare", help="Compare any two standard trainer artifact directories")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--before-label", default="Before")
    compare.add_argument("--after-label", default="After")
    return parser


def doctor() -> int:
    """Print a JSON environment and accelerator readiness report."""
    report = {
        "llmdemo": __version__,
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "gloo_available": torch.distributed.is_gloo_available(),
        "nccl_available": torch.distributed.is_nccl_available(),
        "nvidia_smi": shutil.which("nvidia-smi"),
    }
    print(json.dumps(report, indent=2))
    return 0


def init_project(directory: str, force: bool) -> int:
    """Create a starter YAML project, optionally replacing its config.

    Example:
        ``init_project("my-run", force=False)`` creates
        ``my-run/config.yaml``.
    """
    destination = Path(directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "config.yaml"
    if target.exists() and not force:
        raise SystemExit(f"{target} already exists; use --force to replace it")
    template = files("llmdemo_cli.templates").joinpath("config.yaml")
    target.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Created {target}")
    print(f"Run: llmdemo train --config {target} --device cpu")
    return 0


def train(config_path: str, device: str, processes: int, overrides: list[str]) -> int:
    """Run locally or launch multi-GPU workers through ``torchrun``.

    Example:
        ``train("config.yaml", "cpu", 1, ["trainer.max_epochs=1"])``
    """
    if processes < 1:
        raise SystemExit("--num-processes must be at least 1")
    if processes > 1 and device != "cuda":
        raise SystemExit("--num-processes > 1 currently requires --device cuda")
    os.environ["LLMDEMO_DEVICE"] = device
    if processes == 1:
        run(load_config(config_path, overrides))
        return 0

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={processes}",
        "-m",
        "generic_trainer.cli",
        "--config",
        str(Path(config_path).resolve()),
    ]
    for override in overrides:
        command.extend(["--set", override])
    return subprocess.run(command, check=False, env=os.environ.copy()).returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch CLI arguments and return a process exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "optimizers":
        print("\n".join(sorted(CORE_OPTIMIZERS) + ["distributed_shampoo (optional)"]))
        return 0
    if args.command == "init":
        return init_project(args.directory, args.force)
    if args.command == "train":
        return train(args.config, args.device, args.num_processes, args.set)
    if args.command == "compare":
        from generic_trainer.comparison import compare_runs

        compare_runs(args.before, args.after, args.output, args.before_label, args.after_label)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
