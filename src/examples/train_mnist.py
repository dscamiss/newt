"""
Simple MNIST example.

Based on https://github.com/pytorch/examples/blob/main/mnist/main.py
"""

# flake8: noqa=D102
# pylint: disable=missing-function-docstring

import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Normalize, ToTensor

from newt import Newt, NewtConfig


class ConvNet(nn.Module):
    """Simple convolutional network."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(
    device: Union[str, torch.device],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data_loader: DataLoader,
    epoch: int,
) -> None:
    log_interval = 100
    loss_criterion = torch.nn.NLLLoss()

    # Prepare `model` for training
    model.to(device)
    model.train()

    # Create Newt LR scheduler
    newt_config = NewtConfig(model=model, loss_criterion=loss_criterion)
    newt = Newt(optimizer, newt_config)

    # Run training loop
    for batch_idx, (x, y) in enumerate(train_data_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        newt.step_setup(loss, x, y)
        newt.step()

        if batch_idx % log_interval == 0:
            n = batch_idx * len(x)
            n_total = len(train_data_loader.dataset)
            percent = 100.0 * batch_idx / len(train_data_loader)
            lr = newt.get_last_lr()[0]
            print(
                f"train epoch: {epoch:3d} [{n}/{n_total} ({percent:.2f}%)]\tloss: {loss.item():.4f}\tlr:{lr:.7f}"
            )


def main() -> None:
    """Run MNIST demo."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    data_dir = Path(__file__).resolve().parent / "data"

    transform = transforms.Compose(
        [
            ToTensor(),
            Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
    )

    model = ConvNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        train(device, model, optimizer, train_data_loader, epoch)


def set_seed(seed: int) -> None:
    """Set random seeds etc. to ensure repeatability."""
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Avoid non-deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(1337)
    main()
