from __future__ import annotations

import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .distributed import Runtime, reduce_mean
from .diagnostics import RankDiagnosticLogger
from .logging import MetricLogger
from .optimizers import build_optimizer
from .telemetry import HardwareTelemetry, model_size


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


class Trainer:
    """Generic engine for any nn.Module returning ``{'loss': scalar}``."""

    def __init__(self, model: nn.Module, train_data: Dataset, val_data: Dataset, config: dict[str, Any], runtime: Runtime) -> None:
        self.cfg, self.runtime = config, runtime
        self.tcfg = config["trainer"]
        seed_everything(int(config["run"]["seed"]), runtime.rank)
        self.output_dir = Path(config["run"]["output_dir"])
        self.logger = MetricLogger(self.output_dir, runtime.is_main)
        diagnostics_cfg = self.tcfg.get("diagnostics", {})
        self.diagnostics = RankDiagnosticLogger(self.output_dir, runtime, bool(diagnostics_cfg.get("enabled", False)))
        self.diagnostic_every = int(diagnostics_cfg.get("log_every_batches", 1))
        self.diagnostics.emit("trainer_init_start")
        telemetry_cfg = self.tcfg.get("telemetry", {})
        self.telemetry = HardwareTelemetry(runtime, enabled=bool(telemetry_cfg.get("enabled", True)))
        self.telemetry_every = int(telemetry_cfg.get("log_every_steps", self.tcfg.get("log_every_steps", 20)))
        size = model_size(model)
        self.diagnostics.emit("model_to_device_start", **size)
        self.model = model.to(runtime.device)
        self.diagnostics.emit("model_to_device_complete")
        if self.tcfg.get("compile", False) and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        if runtime.distributed:
            self.diagnostics.emit("ddp_wrap_start")
            kwargs = {"device_ids": [runtime.local_rank], "output_device": runtime.local_rank} if runtime.device.type == "cuda" else {}
            self.model = DistributedDataParallel(self.model, **kwargs)
            self.diagnostics.emit("ddp_wrap_complete")
            self._sync("all_ranks_joined_after_ddp")
        placement = self.telemetry.placement_event(size)
        if placement:
            self.logger.emit(placement)
        self.train_sampler = DistributedSampler(train_data, shuffle=True) if runtime.distributed else None
        self.val_sampler = DistributedSampler(val_data, shuffle=False) if runtime.distributed else None
        common = {"num_workers": int(self.tcfg.get("num_workers", 0)), "pin_memory": runtime.device.type == "cuda"}
        self.train_loader = DataLoader(train_data, batch_size=int(self.tcfg["train_batch_size"]), sampler=self.train_sampler,
                                       shuffle=self.train_sampler is None, **common)
        self.val_loader = DataLoader(val_data, batch_size=int(self.tcfg.get("eval_batch_size", self.tcfg["train_batch_size"])),
                                     sampler=self.val_sampler, shuffle=False, **common)
        self.optimizer, self.optimizer_name = build_optimizer(self.model, self.tcfg)
        self.base_lr = float(self.optimizer.param_groups[0]["lr"])
        self.diagnostics.emit("optimizer_created", optimizer=self.optimizer_name, parameters=self.tcfg.get("optimizer", {}))
        self.global_step, self.start_epoch = 0, 0
        self.best_metric = math.inf if self.tcfg["early_stopping"]["mode"] == "min" else -math.inf
        self.bad_epochs = 0
        precision = self.tcfg.get("precision", "auto")
        if precision == "auto":
            precision = "bf16" if runtime.device.type == "cuda" and torch.cuda.is_bf16_supported() else ("fp16" if runtime.device.type == "cuda" else "fp32")
        self.precision = precision
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.use_amp = precision in {"bf16", "fp16"} and runtime.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and precision == "fp16")
        resume = self.tcfg.get("resume_from")
        if resume:
            self._load_checkpoint(Path(resume))
        self.diagnostics.emit("trainer_init_complete", train_batches=len(self.train_loader), validation_batches=len(self.val_loader))

    def _sync(self, name: str) -> None:
        if not self.runtime.distributed:
            self.diagnostics.emit(name, collective="barrier", status="single_rank")
            return
        started = time.perf_counter()
        self.diagnostics.emit(name, collective="barrier", status="start")
        dist.barrier()
        self.diagnostics.emit(name, collective="barrier", status="complete", duration_seconds=time.perf_counter() - started)

    def _reduce_metric(self, value: float, name: str) -> float:
        started = time.perf_counter()
        self.diagnostics.emit("collective_all_reduce_start", metric=name, local_value=value)
        result = reduce_mean(value, self.runtime)
        self.diagnostics.emit("collective_all_reduce_complete", metric=name, reduced_value=result,
                              duration_seconds=time.perf_counter() - started)
        return result

    def _move(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(self.runtime.device, non_blocking=True)
        if isinstance(value, dict):
            return {k: self._move(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._move(v) for v in value)
        return value

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()

    def _schedule(self, step: int, total_steps: int) -> float:
        warmup = int(self.tcfg.get("warmup_steps", 0))
        if warmup and step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    def fit(self) -> dict[str, float]:
        accumulation = int(self.tcfg.get("gradient_accumulation_steps", 1))
        total_steps = math.ceil(len(self.train_loader) / accumulation) * int(self.tcfg["max_epochs"])
        base_lr = self.base_lr
        summary: dict[str, float] = {}
        try:
            for epoch in range(self.start_epoch, int(self.tcfg["max_epochs"])):
                self.diagnostics.emit("epoch_start", epoch=epoch + 1)
                if self.train_sampler:
                    self.train_sampler.set_epoch(epoch)
                train_loss = self._train_epoch(epoch, accumulation, total_steps, base_lr)
                val_loss = self.evaluate()
                summary = {"train_loss": train_loss, "val_loss": val_loss}
                self.logger.emit({"event": "epoch", "epoch": epoch + 1, "global_step": self.global_step, **summary})
                self.diagnostics.emit("epoch_complete", epoch=epoch + 1, **summary)
                improved = self._is_improved(val_loss)
                self.bad_epochs = 0 if improved else self.bad_epochs + 1
                if improved:
                    self.best_metric = val_loss
                    self._save_checkpoint("best.pt", epoch + 1)
                if (epoch + 1) % int(self.tcfg.get("save_every_epochs", 1)) == 0:
                    self._save_checkpoint("last.pt", epoch + 1)
                if self.bad_epochs >= int(self.tcfg["early_stopping"]["patience"]):
                    self.logger.emit({"event": "early_stop", "epoch": epoch + 1, "val_loss": val_loss})
                    break
            self._sync("all_ranks_joined_training_complete")
            self.diagnostics.emit("training_complete", **summary)
            return summary
        except Exception as error:
            self.diagnostics.emit("training_failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            self.logger.close()

    def _train_epoch(self, epoch: int, accumulation: int, total_steps: int, base_lr: float) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        running, count, skips = 0.0, 0, 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(self.train_loader):
            trace_batch = self.diagnostic_every > 0 and batch_index % self.diagnostic_every == 0
            if trace_batch:
                self.diagnostics.emit("batch_start", epoch=epoch + 1, batch=batch_index + 1)
            should_sync = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(self.train_loader)
            sync_context = nullcontext() if should_sync or not isinstance(self.model, DistributedDataParallel) else self.model.no_sync()
            with sync_context, self._autocast():
                forward_started = time.perf_counter()
                if trace_batch:
                    self.diagnostics.emit("forward_start", epoch=epoch + 1, batch=batch_index + 1)
                output = self.model(**self._move(batch))
                if trace_batch:
                    self.diagnostics.emit("forward_complete", epoch=epoch + 1, batch=batch_index + 1,
                                          duration_seconds=time.perf_counter() - forward_started)
                if not isinstance(output, dict) or "loss" not in output or output["loss"].ndim != 0:
                    raise TypeError("model must return a dict containing scalar 'loss'")
                loss = output["loss"]
            loss_value = float(loss.detach())
            safety = self.tcfg["safety"]
            invalid = not math.isfinite(loss_value) or loss_value > float(safety["max_loss"])
            if invalid:
                self.optimizer.zero_grad(set_to_none=True)
                skips += 1
                self.logger.emit({"event": "unsafe_loss", "epoch": epoch + 1, "loss": loss_value, "skips": skips})
                if safety["stop_on_non_finite_loss"] or skips > int(safety["max_consecutive_skips"]):
                    raise FloatingPointError(f"loss safety circuit breaker triggered: {loss_value}")
                continue
            backward_started = time.perf_counter()
            if trace_batch:
                self.diagnostics.emit("backward_start", epoch=epoch + 1, batch=batch_index + 1)
            self.scaler.scale(loss / accumulation).backward()
            if trace_batch:
                self.diagnostics.emit("backward_complete", epoch=epoch + 1, batch=batch_index + 1,
                                      duration_seconds=time.perf_counter() - backward_started)
            running, count = running + loss_value, count + 1
            if should_sync:
                optimizer_started = time.perf_counter()
                if trace_batch:
                    self.diagnostics.emit("optimizer_step_start", epoch=epoch + 1, batch=batch_index + 1)
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.tcfg["max_grad_norm"]))
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm: {float(grad_norm)}")
                lr = base_lr * self._schedule(self.global_step, total_steps)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if trace_batch:
                    self.diagnostics.emit("optimizer_step_complete", epoch=epoch + 1, batch=batch_index + 1,
                                          global_step=self.global_step, duration_seconds=time.perf_counter() - optimizer_started)
                if self.global_step % int(self.tcfg["log_every_steps"]) == 0:
                    self.logger.emit({"event": "step", "epoch": epoch + 1, "global_step": self.global_step,
                                      "train_loss": loss_value, "grad_norm": float(grad_norm), "learning_rate": lr,
                                      "examples_per_second": count * int(self.tcfg["train_batch_size"]) * self.runtime.world_size / max(1e-6, time.perf_counter() - started)})
                if self.telemetry_every > 0 and self.global_step % self.telemetry_every == 0:
                    utilization = self.telemetry.utilization_event(self.global_step)
                    if utilization:
                        self.logger.emit(utilization)
        return self._reduce_metric(running / max(1, count), "train_loss")

    @torch.no_grad()
    def evaluate(self) -> float:
        self.diagnostics.emit("evaluation_start")
        self.model.eval()
        total, count = 0.0, 0
        for batch in self.val_loader:
            with self._autocast():
                loss = self.model(**self._move(batch))["loss"]
            total, count = total + float(loss), count + 1
        result = self._reduce_metric(total / max(1, count), "val_loss")
        self.diagnostics.emit("evaluation_complete", val_loss=result)
        return result

    def _is_improved(self, value: float) -> bool:
        cfg = self.tcfg["early_stopping"]
        delta = float(cfg.get("min_delta", 0.0))
        return value < self.best_metric - delta if cfg["mode"] == "min" else value > self.best_metric + delta

    def _bare_model(self) -> nn.Module:
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        return getattr(model, "_orig_mod", model)

    def _save_checkpoint(self, name: str, epoch: int) -> None:
        if not self.runtime.is_main:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics.emit("checkpoint_save_start", checkpoint=name, epoch=epoch)
        torch.save({"model": self._bare_model().state_dict(), "optimizer": self.optimizer.state_dict(), "epoch": epoch,
                    "global_step": self.global_step, "best_metric": self.best_metric, "config": self.cfg}, self.output_dir / name)
        self.diagnostics.emit("checkpoint_save_complete", checkpoint=name, epoch=epoch)

    def _load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.runtime.device, weights_only=False)
        self._bare_model().load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch, self.global_step = int(checkpoint["epoch"]), int(checkpoint["global_step"])
        self.best_metric = float(checkpoint["best_metric"])
