"""BEFORE: lightweight model and training infrastructure in one file.

Example:
    ``python -m models.simple_helloworld.before.helloworld`` trains the
    original 128-example, two-epoch demonstration on CPU by default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter


class ToyModel(nn.Module):
    """Fixed 10-to-5 regression network embedded in the coupled program."""

    def __init__(self) -> None:
        """Create two linear layers separated by ReLU."""
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return regression predictions for one batch."""
        return self.net2(self.relu(self.net1(inputs)))


class ToyDataset(Dataset):
    """Small deterministic synthetic regression dataset."""

    def __init__(self, size: int, seed: int = 42) -> None:
        """Generate inputs and labels from a fixed target projection."""
        inputs_generator = torch.Generator().manual_seed(seed)
        projection_generator = torch.Generator().manual_seed(10_000)
        noise_generator = torch.Generator().manual_seed(seed + 20_000)
        self.inputs = torch.randn(size, 10, generator=inputs_generator)
        projection = torch.randn(10, 5, generator=projection_generator)
        self.labels = self.inputs @ projection + 0.05 * torch.randn(size, 5, generator=noise_generator)

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.inputs)

    def __getitem__(self, index: int):
        """Return one input and label tuple."""
        return self.inputs[index], self.labels[index]


def initialize_distributed() -> tuple[torch.device, int, int, bool]:
    """Select CPU or CUDA and initialize a process group when launched by torchrun."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device, backend = torch.device("cpu"), "gloo"
    if distributed:
        dist.init_process_group(backend=backend, init_method="env://")
    return device, rank, local_rank, distributed


def train(epochs: int, examples: int, batch_size: int, output_dir: Path) -> None:
    """Run the intentionally coupled lightweight training loop."""
    device, rank, local_rank, distributed = initialize_distributed()
    torch.manual_seed(42 + rank)
    validation_size = max(1, round(examples * 0.2))
    train_data = ToyDataset(examples - validation_size, seed=42)
    validation_data = ToyDataset(validation_size, seed=43)
    sampler = DistributedSampler(train_data, shuffle=True) if distributed else None
    loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, shuffle=sampler is None)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    model = ToyModel().to(device)
    if distributed:
        options = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        model = DDP(model, **options)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None

    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        train_loss = total / max(1, len(loader))
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                validation_loss += float(loss_fn(model(inputs.to(device)), labels.to(device)))
        validation_loss /= max(1, len(validation_loader))
        if rank == 0:
            event = {"event": "epoch", "epoch": epoch + 1, "global_step": (epoch + 1) * len(loader),
                     "train_loss": train_loss, "val_loss": validation_loss, "device": str(device)}
            print(json.dumps(event), flush=True)
            writer.add_scalar("train_loss", train_loss, epoch + 1)
            writer.add_scalar("val_loss", validation_loss, epoch + 1)
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({"model": state, "epoch": epoch + 1}, output_dir / "last.pt")
    if writer:
        writer.close()
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    """Parse the original lightweight defaults and start training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/simple-helloworld-before"))
    args = parser.parse_args()
    train(args.epochs, args.examples, args.batch_size, args.output_dir)


if __name__ == "__main__":
    main()
