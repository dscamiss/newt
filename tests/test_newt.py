"""Test code for `Newt` class."""

# pylint: disable=not-callable,protected-access

import pytest
import torch
from torch import Tensor, nn

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
    expected_inner_product = torch.tensor(0.0).requires_grad_(False)
    expected_inner_product.add_(torch.norm(weight_update) ** 2.0)  # fmt: skip
    expected_inner_product.add_(torch.norm(bias_update) ** 2.0)  # fmt: skip

    # Check actual versus expected
    err_str = "Error in inner product"
    assert torch.allclose(inner_product, expected_inner_product), err_str


@pytest.mark.parametrize("use_alternate_approx", ["True", "False"])
def test_compute_next_lr(affine_model: nn.Module, use_alternate_approx: bool) -> None:
    """Test `_compute_next_lr()` behavior."""
    # Note the use of `lr=1.0` here
    optimizer = torch.optim.SGD(affine_model.parameters(), lr=1.0)
    config = NewtConfig(
        use_alternate_approx=use_alternate_approx,
        gamma=1.0,
        epsilon=0.0,
    )
    newt = Newt(optimizer, config)

    # Force parameter updates
    weight_update = torch.randn_like(affine_model.weight)
    bias_update = torch.randn_like(affine_model.bias)
    affine_model.weight.data -= weight_update
    affine_model.bias.data -= bias_update

    # Force losses
    newt._curr_loss = torch.tensor(3.0)
    newt._lookahead_loss = torch.tensor(1.0)

    # Force inner products
    newt._curr_inner_product = torch.tensor(2.0)
    newt._lookahead_inner_product = torch.tensor(1.0)

    # Force current gradients
    affine_model.weight.grad = weight_update
    affine_model.bias.grad = bias_update

    # Compute actual and expected learning rates
    lr = newt._compute_next_lr()
    if not use_alternate_approx:
        expected_lr = torch.tensor(2.0)
    else:
        expected_lr = torch.tensor(1.5)

    # Compare actual and expected learning rates
    assert torch.all(lr == expected_lr), "Error in learning rates"


def test_step_setup(
    affine_model: nn.Module, batch_dim: int, input_dim: int, output_dim: int
) -> None:
    """Test `step_setup()` behavior."""
    optimizer = torch.optim.SGD(affine_model.parameters())
    loss_criterion = torch.nn.MSELoss()
    newt_config = NewtConfig(model=affine_model, loss_criterion=loss_criterion)
    newt = Newt(optimizer, newt_config)

    x = torch.randn(batch_dim, input_dim)
    y = torch.randn(batch_dim, output_dim)

    optimizer.zero_grad()
    curr_loss = torch.tensor(1.0)
    lookahead_loss = loss_criterion(affine_model(x), y)
    lookahead_loss.backward()

    grad_cache: dict[nn.Parameter, Tensor] = {}
    for param in affine_model.parameters():
        grad_cache[param] = param.grad.clone().detach()

    newt.step_setup(curr_loss, x, y)

    # Check if `step_setup()` sets correct loss and gradient values
    assert newt._curr_loss == curr_loss, "Error in current loss"
    assert newt._lookahead_loss == lookahead_loss, "Error in next loss"
    for param in affine_model.parameters():
        assert torch.all(param.grad == grad_cache[param]), "Error in gradients"
