"""Test configuration."""

import pytest
import torch
from torch import nn


@pytest.fixture(name="batch_dim")
def fixture_batch_dim() -> int:
    """Batch dimension."""
    return 8


@pytest.fixture(name="input_dim")
def fixture_input_dim() -> int:
    """Input dimension."""
    return 2


@pytest.fixture(name="output_dim")
def fixture_output_dim() -> int:
    """Output dimension."""
    return 3


@pytest.fixture(name="affine_model")
def fixture_affine_model(input_dim: int, output_dim: int) -> nn.Module:
    """Make affine model."""
    return torch.nn.Linear(input_dim, output_dim, bias=True)
