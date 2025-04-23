"""Test code for `Newt` class."""

# pylint: disable=protected-access

import pytest
import torch
from torch import nn

from newt import Newt, NewtConfig


def test_newt_multiple_parameter_groups(affine_model: nn.Module) -> None:
    """Test constructor with multiple parameter groups."""
    param_groups = [
        {"params": affine_model.weight},
        {"params": affine_model.bias},
    ]
    optimizer = torch.optim.SGD(param_groups)
    with pytest.raises(ValueError):
        Newt(optimizer, NewtConfig())


def test_newt_differentiable_optimizer(affine_model: nn.Module) -> None:
    """Test constructor with multiple parameter groups."""
    optimizer = torch.optim.SGD(affine_model.parameters(), differentiable=True)
    with pytest.raises(ValueError):
        Newt(optimizer, NewtConfig())


def test_refresh_param_cache_no_frozen_params(affine_model: nn.Module) -> None:
    """Test `_refresh_param_cache()` with no frozen parameters."""
    optimizer = torch.optim.SGD(affine_model.parameters(), lr=1e-3)
    newt = Newt(optimizer, NewtConfig())
    newt._refresh_param_cache()

    err_str = "Unexpected parameter cache state"
    for param in affine_model.parameters():
        assert param in newt._param_cache, err_str


def test_refresh_param_cache_frozen_params(affine_model: nn.Module) -> None:
    """Test `_refresh_param_cache()` with frozen parameters."""
    affine_model.bias.requires_grad_(False)
    optimizer = torch.optim.SGD(affine_model.parameters(), lr=1e-3)
    newt = Newt(optimizer, NewtConfig())
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
    newt = Newt(optimizer, NewtConfig())

    newt._param_cache[affine_model.bias] = None  # Frozen and included
    with pytest.raises(RuntimeError):
        newt._refresh_param_cache()


def test_get_param_update(affine_model: nn.Module) -> None:
    """Test `_get_param_update()` behavior."""
    # Note the use of `lr=1.0` here
    optimizer = torch.optim.SGD(affine_model.parameters(), lr=1.0)
    config = NewtConfig(cache_updates=True)
    newt = Newt(optimizer, config)

    # Force parameter updates
    expected_weight_update = torch.randn_like(affine_model.weight)
    expected_bias_update = torch.randn_like(affine_model.bias)
    affine_model.weight.data -= expected_weight_update
    affine_model.bias.data -= expected_bias_update

    # Compute actual weight updates
    weight_update = newt._get_param_update(affine_model.weight)
    bias_update = newt._get_param_update(affine_model.bias)

    # Check actual versus expected
    assert torch.allclose(weight_update, expected_weight_update), "Error in weight update"
    assert torch.allclose(bias_update, expected_bias_update), "Error in bias update"


def test_compute_inner_product(affine_model: nn.Module) -> None:
    """Test `_compute_inner_product()` behavior."""
    # Note the use of `lr=1.0` here
    optimizer = torch.optim.SGD(affine_model.parameters(), lr=1.0)
    config = NewtConfig(cache_updates=True)
    newt = Newt(optimizer, config)

    # Force parameter updates
    weight_update = torch.randn_like(affine_model.weight)
    bias_update = torch.randn_like(affine_model.bias)
    affine_model.weight.data -= weight_update
    affine_model.bias.data -= bias_update

    # Force "next loss" gradients
    affine_model.weight.grad = weight_update
    affine_model.bias.grad = bias_update

    # Compute actual and expected inner products
    inner_product = newt._compute_inner_product()
    expected_inner_product = torch.as_tensor(0.0).requires_grad_(False)
    expected_inner_product.add_(torch.norm(weight_update) ** 2.0)  # fmt: skip
    expected_inner_product.add_(torch.norm(bias_update) ** 2.0)  # fmt: skip

    # Check actual versus expected
    err_str = "Error in inner product"
    assert torch.allclose(inner_product, expected_inner_product), err_str


def test_compute_next_lr(affine_model: nn.Module) -> None:
    """Test `_compute_next_lr()` behavior."""
    # Note the use of `lr=1.0` here
    optimizer = torch.optim.SGD(affine_model.parameters(), lr=1.0)
    config = NewtConfig(gamma=1.0)
    newt = Newt(optimizer, config)

    # Force parameter updates
    weight_update = torch.randn_like(affine_model.weight)
    bias_update = torch.randn_like(affine_model.bias)
    affine_model.weight.data -= weight_update
    affine_model.bias.data -= bias_update

    # Force "next loss" gradients
    affine_model.weight.grad = weight_update
    affine_model.bias.grad = bias_update

    # Force loss and "next loss"
    newt._curr_loss = 2.0
    newt._next_loss = 1.0

    # Compute actual and expected learning rates
    lr = newt._compute_next_lr()
    inner_product = newt._compute_inner_product()
    expected_lr_num = inner_product
    expected_lr_den = 2.0 * (1.0 - inner_product)
    expected_lr = 1.0 + (expected_lr_num / (newt._config.epsilon + expected_lr_den))

    # Check actual versus expected
    assert torch.allclose(lr, expected_lr), "Error in learning rate"
