"""
Simple MNIST example.

Based on https://github.com/pytorch/examples/blob/main/mnist/main.py
"""

# flake8: noqa=D102
# pylint: disable=missing-function-docstring,not-callable

import random
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
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
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(
    device: Union[str, torch.device],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data_loader: DataLoader,
    epoch: int,
    loss_history: list,
    lr_history: list,
) -> None:
    log_interval = 100
    loss_criterion = torch.nn.NLLLoss()
    n_offset = lr_history[-1][0] if lr_history else 0

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
            loss = loss.item()
            lr = newt.get_last_lr()[0].item()

            print(
                f"train epoch: {epoch:3d} \t"
                f"[{n}/{n_total} ({percent:.2f}%)] \t"
                f"loss: {loss:.4f}\tlr:{lr:.7f}"
            )

            loss_history.append((n_offset + n, loss))
            lr_history.append((n_offset + n, lr))


def main() -> None:
    """Run MNIST demo."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    parent_dir = Path(__file__).resolve().parent
    data_dir = parent_dir / "data"
    plots_dir = parent_dir / "plots"

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
        batch_size=128,
        shuffle=True,
    )

    model = ConvNet()
    model_state_dict = model.state_dict()
    optimizers = [
        ("SGD", torch.optim.SGD),
        ("Adam", torch.optim.Adam),
        ("AdamW", torch.optim.AdamW),
    ]
    num_epochs = 10

    for optimizer_tag, optimizer_type in optimizers:
        model = ConvNet()
        model.load_state_dict(model_state_dict)
        optimizer = optimizer_type(model.parameters())
        
        loss_history = []
        lr_history = []

        for epoch in range(1, num_epochs + 1):
            train(device, model, optimizer, train_data_loader, epoch, loss_history, lr_history)
    
        loss_history = np.array(loss_history)
        lr_history = np.array(lr_history)
    
        _, axes = plt.subplots(1, 2)
        axes[0].plot(loss_history[:, 0], loss_history[:, 1], linewidth=2)
        axes[0].set_xlabel("batch number")
        axes[0].set_ylabel("loss")
        axes[0].grid(True)
        axes[1].plot(lr_history[:, 0], lr_history[:, 1], linewidth=2)
        axes[1].set_xlabel("batch number")
        axes[1].set_ylabel("learning rate")
        axes[1].grid(True)
        plt.suptitle(f"MNIST example - {optimizer_tag}")
        plt.tight_layout()
        plt.show()
    
        plots_dir.mkdir(parents=True, exist_ok=True)
        png_filename = f"train_mnist_{optimizer_tag}.png"
        plt.savefig(plots_dir / png_filename)


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
