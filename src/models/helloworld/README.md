# Hello World: coupled versus decoupled training

This example turns the photographed `hello_world.py` pattern into a runnable
leadership demonstration.

## Before: one file owns everything

[`before/helloworld.py`](before/helloworld.py) contains:

- `ToyModel`
- synthetic data
- device and backend selection
- process-group initialization
- DDP wrapping
- loss and optimizer
- DataLoader and epoch loop
- TensorBoard logging
- checkpoint writing
- command-line parsing

An infrastructure optimization therefore requires editing this model program,
then repeating that work for every other model.

## After: `helloworld.py` is only model code

[`new/helloworld.py`](new/helloworld.py) imports only PyTorch model APIs. It has
no DDP, CUDA, optimizer, DataLoader, TensorBoard, checkpoint, or epoch-loop
logic. [`new/data.py`](new/data.py) owns the domain batch contract, and the very
small [`new/factory.py`](new/factory.py) connects configuration to those domain
objects. The shared `generic_trainer` owns all infrastructure behavior.

## Run the comparison

From the LLMDemo root:

```bash
./src/models/helloworld/run_comparison.sh
```

The default run uses **1,000,000 deterministic examples**, five epochs, and a
batch size of 4,096. Both programs receive the same train/validation data
contract. For a quick laptop rehearsal:

```bash
NUM_EXAMPLES=10000 EPOCHS=2 BATCH_SIZE=512 \
  ./src/models/helloworld/run_comparison.sh
```

Every invocation creates a timestamped directory under
`artifacts/helloworld-comparison/` containing:

- `before/` and `after/` checkpoints, JSONL metrics, and TensorBoard events;
- `visuals/leadership_comparison.png`, a four-panel leadership dashboard;
- `visuals/comparison_report.json`, independent MSE, prediction difference,
  and checkpoint weight/bias norms.

The dashboard compares learning curves, independent held-out quality,
predictions versus expected values, and checkpoint parameter norms. The two
checkpoints are not expected to be byte-identical: the centralized trainer
adds configurable scheduling, clipping, validation, safety, and checkpoint policy. The
comparison demonstrates that model behavior remains comparable while the
infrastructure implementation moves out of model code.

The checked-in after configuration selects `scheduler.name: constant` to
match the constant `0.01` learning rate hard-coded in the before program. A
team can instead select `cosine` or `linear` without editing model Python.

To inspect both runs interactively, use the exact TensorBoard command printed
by the script, for example:

```bash
.venv/bin/tensorboard \
  --logdir artifacts/helloworld-comparison/20261106-120000 --port 6006
```

Then open `http://localhost:6006`. Select `before` and `after` to overlay
`train_loss` and `val_loss`.

Or run each side separately:

```bash
# Before, CPU
LLMDEMO_DEVICE=cpu .venv/bin/python -m models.helloworld.before.helloworld \
  --examples 1000000 --batch-size 4096 --epochs 5

# After, CPU
LLMDEMO_DEVICE=cpu .venv/bin/python -m generic_trainer.cli \
  --config src/models/helloworld/new/config.yaml

# Before, two NVIDIA GPUs
.venv/bin/torchrun --standalone --nproc_per_node=2 \
  -m models.helloworld.before.helloworld

# After, the same model on two NVIDIA GPUs without changing model code
LLMDEMO_DEVICE=cuda .venv/bin/torchrun --standalone --nproc_per_node=2 \
  -m generic_trainer.cli --config src/models/helloworld/new/config.yaml
```

The leadership point is visible in the source diff: infrastructure policy is
deleted from the model program and centralized once in `generic_trainer`.
