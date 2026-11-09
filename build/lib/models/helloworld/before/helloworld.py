"""BEFORE: model code and training infrastructure intertwined in one file.

This intentionally mirrors the pattern in the supplied hello_world.py photos.
It is runnable, but every model would have to copy and maintain its own DDP,
device, optimizer, logging, checkpointing, and training-loop implementation.
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


# ----------------------------- MODEL CODE ---------------------------------
class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net2(self.relu(self.net1(inputs)))


class ToyDataset(Dataset):
    def __init__(self, size: int, seed: int = 42) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.inputs = torch.randn(size, 10, generator=generator)
        projection = torch.randn(10, 5, generator=generator)
        self.labels = self.inputs @ projection + 0.05 * torch.randn(size, 5, generator=generator)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index], self.labels[index]


# -------------------------- INFRASTRUCTURE CODE ----------------------------
def initialize_distributed() -> tuple[torch.device, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if distributed:
        dist.init_process_group(backend=backend, init_method="env://")
    return device, rank, local_rank, distributed


def train(epochs: int, examples: int, batch_size: int, output_dir: Path) -> None:
    device, rank, local_rank, distributed = initialize_distributed()
    dataset = ToyDataset(examples)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler is None)

    model = ToyModel().to(device)
    if distributed:
        ddp_options = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        model = DDP(model, **ddp_options)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / max(1, len(loader))
        if rank == 0:
            event = {"epoch": epoch + 1, "loss": epoch_loss, "device": str(device), "world_size": int(os.environ.get("WORLD_SIZE", "1"))}
            print(json.dumps(event), flush=True)
            writer.add_scalar("Loss/train", epoch_loss, epoch + 1)
            torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(), output_dir / "last.pt")

    if writer:
        writer.close()
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/helloworld-before"))
    args = parser.parse_args()
    train(args.epochs, args.examples, args.batch_size, args.output_dir)


if __name__ == "__main__":
    main()
