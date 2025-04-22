"""Test code for `Newt` class."""

# pylint: disable=protected-access

import pytest
import torch
from torch import nn

from newt import Newt


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
    """Input dimension."""
    return 3


@pytest.fixture(name="affine_model")
def fixture_affine_model(input_dim: int, output_dim: int) -> nn.Module:
    """Make affine model."""
    return torch.nn.Linear(input_dim, output_dim, bias=True)


def test_newt_multiple_parameter_groups(affine_model: nn.Module) -> None:
    """Test constructor with multiple parameter groups."""
    param_groups = [
        {"params": affine_model.weight},
        {"params": affine_model.bias},
    ]
    optimizer = torch.optim.SGD(param_groups)
    loss_criterion = torch.nn.MSELoss()
    with pytest.raises(ValueError):
        Newt(optimizer, loss_criterion)


def test_newt_differentiable_optimizer(affine_model: nn.Module) -> None:
    """Test constructor with multiple parameter groups."""
    optimizer = torch.optim.SGD(affine_model.parameters(), differentiable=True)
    loss_criterion = torch.nn.MSELoss()
    with pytest.raises(ValueError):
        Newt(optimizer, loss_criterion)


def test_refresh_param_cache_no_frozen_params(affine_model: nn.Module) -> None:
    """Test `_refresh_param_cache()` with no frozen parameters."""
    optimizer = torch.optim.SGD(affine_model.parameters(), lr=1e-3)
    loss_criterion = torch.nn.MSELoss()
    newt = Newt(optimizer, loss_criterion)
    newt._refresh_param_cache()

    err_str = "Unexpected parameter cache state"
    for param in affine_model.parameters():
        assert param in newt._param_cache, err_str


def test_refresh_param_cache_frozen_params(affine_model: nn.Module) -> None:
    """Test `_refresh_param_cache()` with frozen parameters."""
    affine_model.bias.requires_grad_(False)
    optimizer = torch.optim.SGD(affine_model.parameters(), lr=1e-3)
    loss_criterion = torch.nn.MSELoss()
    newt = Newt(optimizer, loss_criterion)
    newt._refresh_param_cache()

    err_str = "Unexpected parameter cache state"
    for param in affine_model.parameters():
        if not param.requires_grad:
            assert param not in newt._param_cache, err_str
        else:
            assert param in newt._param_cache, err_str


def test_refresh_param_cache_unexpected(affine_model: nn.Module) -> None:
    """Test `_refresh_param_cache()` with unexpected state."""
    affine_model.bias.requires_grad_(False)
    optimizer = torch.optim.SGD(affine_model.parameters(), lr=1e-3)
    loss_criterion = torch.nn.MSELoss()
    newt = Newt(optimizer, loss_criterion)

    newt._param_cache[affine_model.bias] = None  # Frozen and included
    with pytest.raises(RuntimeError):
        newt._refresh_param_cache()


def test_get_param_update(affine_model: nn.Module) -> None:
    """Test update cache behavior."""
    lr = 1.0  # Prevent floating-point errors
    optimizer = torch.optim.SGD(affine_model.parameters(), lr=lr)
    loss_criterion = torch.nn.MSELoss()
    newt = Newt(optimizer, loss_criterion, cache_updates=True)

    weight_delta = torch.randn_like(affine_model.weight)
    bias_delta = torch.randn_like(affine_model.bias)
    affine_model.weight.data -= lr * weight_delta
    affine_model.bias.data -= lr * bias_delta

    weight_update = newt._get_param_update(affine_model.weight)
    bias_update = newt._get_param_update(affine_model.bias)

    assert torch.allclose(weight_update, weight_delta), "Error in weight update"
    assert torch.allclose(bias_update, bias_delta), "Error in bias update"
