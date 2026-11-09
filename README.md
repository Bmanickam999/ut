# AI infrastructure decoupling demo

This repository demonstrates the boundary that is missing in the photographed sample: **model teams own a normal `torch.nn.Module` and data contract; the infrastructure team owns one reusable, configuration-driven trainer.** No training-loop, DDP, device, checkpoint, or logging code lives in the model package.

The example is a decoder-only transformer with a regression head that predicts median home value from public Census-style housing facts. The production profile is approximately 500M parameters. It uses no bank data, bank APIs, or proprietary services. A tiny profile is included so the complete path can be demonstrated on a laptop.

## Repository map

```text
configs/                 YAML: 500M and smoke-test profiles
src/models/housing_llm/  housing model, tokenizer, and public data
src/models/helloworld/   runnable before/after decoupling demonstration
src/generic_trainer/     infra-team boundary (generic PyTorch training engine)
notebooks/               separate training and inference examples
tests/                   fast contract tests
```

The complete trainer architecture, contracts, YAML specification, failure
model, framework comparison, and roadmap are documented in
[`src/generic_trainer/README.md`](src/generic_trainer/README.md).

## Generic Trainer documentation

- **[Open the Generic Trainer specification and architecture guide](src/generic_trainer/README.md)**
- [Browse the Generic Trainer Python package](src/generic_trainer/)

These are repository-relative links: they work from a local Markdown viewer,
VS Code, and GitHub after the repository is uploaded or cloned. GitHub will
automatically convert them to the corresponding URLs for the current branch.

### Advanced trainer capabilities

The infrastructure layer supports native DDP and FSDP plus model-specific
tensor/pipeline partitioning through validated strategy factories. Specialized
GAN, reinforcement-learning, multi-optimizer, unbounded-streaming, and elastic
recovery workflows remain outside the current supervised engine. See the
linked Generic Trainer guide for contracts and YAML.

Model parallelism is selected explicitly in each training YAML:

```yaml
trainer:
  model_parallelism:
    name: none                 # none, fsdp, tensor_parallel, pipeline_parallel
    tensor_parallel_size: 1
    pipeline_parallel_size: 1
    data_parallel_size: auto
```

With `name: none`, multi-GPU launches use DDP only; no model parallelism is
enabled. FSDP is built in. Tensor and pipeline modes additionally require a
model-specific `factory` and topology. See the
[copyable parallelism configurations](configs/model_parallelism_examples.yaml).

Distributed diagnostics defaults such as `NCCL_DEBUG=INFO` are also declared
under `trainer.environment.variables`. Existing cluster/shell values win by
default, and the trainer applies effective values before process-group startup.

## Quick start

Python 3.10+ is recommended.

```bash
# Automatic isolated setup (CPU, Apple Silicon, or NVIDIA GPU)
./setup.sh --auto --run-tests

# Fast CPU/single-GPU demonstration
./run_demo.sh --smoke --cpu

# Four ranks on an eight-GPU host; telemetry marks the other four unused
NUM_GPUS=4 ./run_demo.sh --500m --gpu

```

## Recreate `.venv` on an office laptop

Do not copy `.venv` from another machine. Virtual environments contain
absolute paths and platform-specific PyTorch binaries. Clone or extract the
source repository, then recreate the environment locally:

```bash
cd LLMDemo
chmod +x recreate_venv.sh setup.sh run_demo.sh

# Most office laptops without an NVIDIA GPU
./recreate_venv.sh --cpu --run-tests

# Linux workstation with an approved NVIDIA driver/CUDA environment
./recreate_venv.sh --gpu --run-tests

# Automatically select NVIDIA when available, otherwise CPU
./recreate_venv.sh --auto --run-tests
```

The script renames an existing `.venv` to a timestamped backup before creating
the replacement. If installation fails, it restores the previous environment.
After successful setup:

```bash
source .venv/bin/activate
llmdemo doctor
./run_demo.sh --smoke --cpu
```

For a specific approved CUDA wheel:

```bash
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  ./recreate_venv.sh --gpu --run-tests
```

If the office uses a private Python package mirror or TLS inspection, obtain
the approved values from the platform/security team and pass standard pip
environment settings rather than disabling certificate validation:

```bash
export PIP_INDEX_URL="https://pypi.company.example/simple"
export PIP_CERT="/approved/path/company-ca.pem"
export HTTPS_PROXY="http://approved-proxy.company.example:8080"
./recreate_venv.sh --cpu --run-tests
```

Never commit credentials into this repository or use `--trusted-host` to
bypass corporate TLS validation. The script supports macOS, Linux, and WSL.

## Install and use the local CLI

Install `llmdemo` into an isolated user environment. This does not modify the
project `.venv` or the system Python installation.

```bash
cd /Users/sathyasaravaiya/Documents/Codex/LLMDemo

# CPU installation
./install_cli.sh --cpu

# NVIDIA installation; fails if CUDA is not usable
./install_cli.sh --gpu
```

The installer creates `~/.local/bin/llmdemo`. If that directory is not already
on `PATH`, follow the export command printed by the installer.

```bash
# Verify the installation and hardware backends
llmdemo doctor

# Show available YAML optimizers
llmdemo optimizers

# Create a starter configuration in the current directory
llmdemo init my-training-run

# Train locally on CPU
llmdemo train --config my-training-run/config.yaml --device cpu

# Train on one NVIDIA GPU
llmdemo train --config configs/train_500m.yaml --device cuda

# Train with four local CUDA/DDP ranks
llmdemo train --config configs/train_1b.yaml --device cuda --num-processes 4

# Override existing YAML values without editing the file
llmdemo train --config configs/smoke.yaml --device cpu \
  --set trainer.max_epochs=1 --set data.num_examples=64
```

For Meta Distributed Shampoo support:

```bash
./install_cli.sh --gpu --with-shampoo
```

Use `./setup.sh --cpu` to force a CPU wheel or `./setup.sh --gpu` to require
CUDA/NCCL. When a cluster needs a particular CUDA wheel, set the official
PyTorch index explicitly, for example
`PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./setup.sh --gpu`.
The script fails with a clear diagnostic if Python, venv, CUDA, or NCCL is not
usable; it never silently converts a requested GPU setup into CPU execution.

## CPU and GPU setup and run commands

Run all commands from the project directory:

```bash
cd /Users/sathyasaravaiya/Documents/Codex/LLMDemo
```

### CPU only

Install the CPU environment once:

```bash
./setup.sh --cpu
```

Run the fast smoke model on CPU:

```bash
./run_demo.sh --smoke --cpu
```

Run the full 498.7M-parameter model on CPU:

```bash
./run_demo.sh --500m --cpu
```

The full 500M CPU run requires substantial RAM and can take considerably longer
than an accelerated run.

### Apple Silicon GPU using MPS

Install the macOS environment once:

```bash
./setup.sh --auto
```

Run the smoke or 500M profile on the Apple GPU:

```bash
./run_demo.sh --smoke --mps
./run_demo.sh --500m --mps
```

### Single NVIDIA GPU using CUDA

Install and verify CUDA-enabled PyTorch and NCCL:

```bash
./setup.sh --gpu
```

Run the smoke or 500M profile on one NVIDIA GPU:

```bash
./run_demo.sh --smoke --gpu
./run_demo.sh --500m --gpu
```

If the cluster requires a specific official PyTorch CUDA wheel, supply its
index during setup:

```bash
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./setup.sh --gpu
```

### Multiple NVIDIA GPUs using DDP

Set `NUM_GPUS` to the number of GPUs that the trainer should use. One process
and one DDP rank are created per selected GPU:

```bash
# Use four GPUs for the 500M model
NUM_GPUS=4 ./run_demo.sh --500m --gpu

# Use all eight GPUs
NUM_GPUS=8 ./run_demo.sh --500m --gpu
```

On an eight-GPU server, the four-GPU command records GPUs 0–3 as assigned and
GPUs 4–7 as unused in `artifacts/housing-llm-500m/metrics.jsonl`, provided all
eight GPUs are visible to `nvidia-smi`.

### Confirm the selected device

The first startup event states the actual device. For example:

```json
{"event": "startup", "device": "cpu", "parameters": 498675713}
```

The value will be `cpu`, `mps`, or `cuda:N`. An explicitly requested device
fails with a clear error if it is unavailable; it does not silently fall back.

## Tests, including DDP and collectives

```bash
# Unit and two-process CPU/Gloo distributed tests
pytest -q

# Only the real collective/DDP suite (CPU tests run; GPU tests skip if unavailable)
pytest -q -m distributed

# On a host with at least two CUDA GPUs, require the NCCL tests to run
pytest -q -m "distributed and gpu" -rs
```

The distributed suite launches two independent processes and verifies SUM
all-reduce, broadcast, all-gather, barrier, the trainer's mean reduction,
DDP gradient averaging, and a complete CPU-DDP training epoch. NCCL tests use
the same assertions on two GPUs. They skip—not silently fall back—when fewer
than two CUDA devices are present.

Training writes rank-zero metrics to `artifacts/*/metrics.jsonl`, TensorBoard events when TensorBoard is installed, and resumable `last.pt`/`best.pt` checkpoints. Configuration can be overridden without changing code:

```bash
python -m generic_trainer.cli --config configs/smoke.yaml \
  --set trainer.max_epochs=2 data.num_examples=1024
```

At startup the trainer emits a `hardware_placement` event containing exact
parameter, buffer, byte, and GiB counts; every rank-to-device assignment; the
full `nvidia-smi` inventory; and `unused_gpu_indices`. During training it emits
`gpu_utilization` events with utilization and memory for every installed GPU,
including GPUs not assigned to a rank. The training notebook converts these
events into rank-labeled utilization and memory graphs for the leadership demo.

## Data

`data.source.type` supports:

- `synthetic`: deterministic Census-shaped demo rows, requiring no network.
- `csv`: a local CSV with columns listed in `data.feature_fields` and `data.target_field`.
- `url_csv`: a public HTTPS CSV, downloaded and cached only when explicitly selected.

The default 500M config intentionally points to the synthetic provider so the demo cannot be blocked by bank technology. To use a Census export, change only the YAML source and column mapping. The loader validates schemes, columns, finite values, and row limits.

## 500M expectation

The production configuration is about 498M trainable parameters (the CLI prints the authoritative count). It can execute on CPU or GPU, but full training is intended for multiple GPUs: fp32 weights alone require about 2 GB, and training needs substantially more for gradients, optimizer state, and activations. Use `configs/smoke.yaml` to prove the architecture and trainer contract on CPU.

## Hardened 1B run and diagnostic logs

`configs/train_1b.yaml` defines a 1,004,794,881-parameter model. It enables a
metadata-only memory preflight before real weights or optimizer state are
allocated. By default, the check estimates 20 bytes per trainable parameter
plus a 4 GiB activation/workspace reserve. The job fails immediately when that
estimate exceeds 90% of available per-rank memory.

```bash
# One sufficiently large NVIDIA GPU
./run_demo.sh --1b --gpu

# Four DDP replicas; each GPU must independently fit one full model replica
NUM_GPUS=4 ./run_demo.sh --1b --gpu
```

The current trainer uses DDP, which replicates the model on every rank. Adding
GPUs improves throughput but does not make an oversized model fit. If the
preflight rejects the 1B model, use a larger-memory accelerator or add
FSDP/ZeRO-style sharding rather than disabling the check.

Detailed diagnostics are controlled in YAML:

```yaml
trainer:
  diagnostics:
    enabled: true
    log_every_batches: 1
    collective_timeout_seconds: 600
    fail_fast_capacity_check: true
    capacity:
      bytes_per_parameter: 20.0
      activation_reserve_gib: 4.0
      max_memory_fraction: 0.9
      require_memory_check: true
```

Each rank writes a flush-on-write JSONL stream:

```text
artifacts/housing-llm-1b/diagnostics/rank-0.jsonl
artifacts/housing-llm-1b/diagnostics/rank-1.jsonl
...
```

Events cover process-group initialization, model placement, DDP wrapping,
all-rank barriers, forward and backward duration, optimizer steps, evaluation,
all-reduce start/completion, checkpoint writes, successful completion, and
failure details. In PyTorch DDP these are worker **processes/ranks**, not
threads; `all_ranks_joined_*` means every rank completed the corresponding
barrier. Per-batch logging is intentionally optional because it adds I/O and
synchronization overhead. Set `log_every_batches: 10` or `100` for sustained
production runs.

Enable diagnostics temporarily on the smoke profile:

```bash
LLMDEMO_DEVICE=cpu .venv/bin/python -m generic_trainer.cli \
  --config configs/smoke.yaml \
  --set trainer.diagnostics.enabled=true \
  --set trainer.diagnostics.log_every_batches=1
```

## YAML-selectable optimizers

The trainer does not hard-code AdamW. Select and configure the optimizer under
`trainer.optimizer`; the trainer validates every argument and records the
resolved optimizer in diagnostics and checkpoints.

Core PyTorch choices requiring no additional packages are `sgd`, `adam`,
`adamw`, `rmsprop`, `adagrad`, `adadelta`, `nadam`, and `radam`.

```yaml
trainer:
  optimizer:
    name: adam
    parameters:
      lr: 0.0003
      betas: [0.9, 0.999]
      eps: 1.0e-8
      weight_decay: 0.01
```

SGD with momentum:

```yaml
trainer:
  optimizer:
    name: sgd
    parameters:
      lr: 0.01
      momentum: 0.9
      nesterov: true
      weight_decay: 0.0001
```

Meta Distributed Shampoo is an optional integration because it is not included
in core PyTorch and has stricter PyTorch, Python, CUDA, compute, and memory
requirements. Install it explicitly:

```bash
./setup.sh --gpu --with-shampoo
```

Then select it in YAML:

```yaml
trainer:
  optimizer:
    name: distributed_shampoo
    parameters:
      lr: 0.001
      betas: [0.9, 0.999]
      epsilon: 1.0e-8
      weight_decay: 0.00001
      max_preconditioner_dim: 8192
      precondition_frequency: 100
    grafting:
      name: adam
      beta2: 0.999
      epsilon: 1.0e-8
```

If Shampoo is selected but its package is absent or incompatible, training
fails before the first batch. The trainer never silently substitutes another
optimizer.

## Generic run comparison

Any two runs that follow the artifact contract (`metrics.jsonl` plus
`last.pt`) can be compared without importing model code:

```bash
llmdemo compare \
  --before artifacts/baseline \
  --after artifacts/candidate \
  --output artifacts/comparison \
  --before-label Baseline \
  --after-label Candidate
```

The command creates `run_comparison.png` and `run_comparison.json` with loss
curves, final training and validation loss, absolute and percentage
differences, `Loss Improved`, and checkpoint tensor norms. This works for
traditional PyTorch models and LLMs as long as both runs emit the standard
trainer artifacts. Model-specific prediction plots can be layered on through
the hello-world evaluator when domain outputs are meaningful.

Two hello-world demonstrations are retained:

- `src/models/simple_helloworld` preserves the original lightweight run with
  128 examples, two epochs, and batch size 16.
- `src/models/helloworld` is the one-million-example leadership comparison.

## Leadership demo narrative

1. Show the model package: it contains domain logic only.
2. Run the same trainer with `smoke.yaml` on CPU.
3. Switch to `train_500m.yaml` or launch with `torchrun`; no model code changes.
4. Point out that mixed precision, DDP, logging, checkpointing, early stopping, gradient clipping, and non-finite-loss protection are all infrastructure policy in YAML/trainer code.
5. Explain that another `nn.Module` can use the same engine by returning a scalar `loss` (or by supplying a small step adapter).

## Safety and scope

This is an educational infrastructure demo, not a valuation product. Synthetic data is not suitable for real decisions. Real deployments need data licenses, privacy review, bias evaluation, drift monitoring, calibration, and model-risk governance.
