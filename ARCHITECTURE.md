# Architecture and ownership contract

## Before: coupled sample

The supplied photographs show model construction, device selection, DDP wrapping, process-group initialization, optimizer choice, synthetic batches, TensorBoard, and the epoch loop in one file. That pattern forces every model team to adopt infrastructure improvements independently.

## After: stable seam

```text
YAML ──────────────┐
                   v
public data -> model factory -> torch.nn.Module
                   |               |
                   └──── loss <────┘
                           |
                           v
                 generic trainer (infra)
          DDP / AMP / safety / logs / checkpoints
```

### Model-team contract

- Supply a `module:function` factory accepting the `model` YAML mapping.
- Return a `torch.nn.Module`.
- Its `forward(**batch)` returns a dictionary with a scalar `loss` during training.
- Supply a data factory returning `(train_dataset, validation_dataset)`.
- Keep domain preprocessing, objectives, and output interpretation here.

### Infrastructure-team contract

- Select device/backend and initialize DDP from `torchrun` environment variables.
- Own batching, optimizer, precision, accumulation, clipping, schedules, logging, checkpoints, resume, early stopping, and safety stops.
- Never import a concrete model package; resolve its factory from configuration.
- Write artifacts only on rank zero.

This boundary means CUDA kernels, compiler options, optimizer implementations, checkpoint formats, or telemetry can be upgraded once without editing model code.

## Parameter budget

The 500M profile uses a 32K vocabulary, width 1024, 37 blocks, 16 attention heads, and a 4096-wide feed-forward layer. With tied token/output embeddings it contains **498,675,713 trainable parameters**. The startup event computes and prints the actual count to prevent configuration drift.

## Production extensions

This demo deliberately stays compact. A production trainer would commonly add FSDP/ZeRO for optimizer-state sharding, fault-tolerant distributed checkpointing, a data catalog, experiment tracking, schema/version lineage, evaluation suites, and policy gates. Those remain trainer/platform changes rather than per-model rewrites.
