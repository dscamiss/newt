"""A Newton-like learning rate scheduler."""

from dataclasses import dataclass
from typing import Optional

import torch
from jaxtyping import Num, jaxtyped
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from typeguard import typechecked as typechecker

from newt.types import LossCriterion, Optimizer, ParamTensorDict


@dataclass
class NewtConfig:
    """
    Dataclass to hold `Newt` configuration.

    Attrs:
        model: Model instance (default = None).
        loss_criterion: Loss criterion instance (default = None).
        gamma: Slow-adaptation parameter for LR updates (default = 0.1).
        epsilon: Constant for divide-by-zero protection (default = 1e-9).
        clamp_min: Minimum value to clamp LR (default = 1e-7).
        clamp_max: Maximum value to clamp LR (default = 1.0).
        cache_updates: Keep a cache of parameter updates (default = False).
            This is only useful for development and testing purposes.
    """

    model: Optional[nn.Module] = None
    loss_criterion: Optional[LossCriterion] = None
    gamma: float = 0.1
    epsilon: float = 1e-9
    clamp_min: float = 1e-7
    clamp_max: float = 1.0
    cache_updates: bool = False


class Newt(LRScheduler):
    """
    The `newt` learning rate scheduler.

    Args:
        optimizer: Optimizer instance.
        config: `Newt` configuration.
        last_epoch: Index of the last epoch (default = -1).
    """

    # TODO: Remove single parameter group limitation.
    def __init__(
        self,
        optimizer: Optimizer,
        config: NewtConfig,
        last_epoch: int = -1,
    ) -> None:
        # Sanity check on number of parameter groups
        if len(optimizer.param_groups) != 1:
            raise ValueError("Optimizer must have exactly one parameter group")

        # Sanity check on differentiable optimizer
        if optimizer.param_groups[0]["differentiable"]:
            raise ValueError("Differentiable optimizers are not supported")

        self._optimizer = optimizer  # TODO: Use superclass `optimizer`?
        self._config = config

        self._curr_loss = None
        self._next_loss = None
        self._param_cache = ParamTensorDict()
        self._update_cache = ParamTensorDict()

        # Initialize parameter cache
        self._refresh_param_cache()

        super().__init__(optimizer, last_epoch)

    def _refresh_param_cache(self) -> None:
        """
        Refresh parameter cache with current parameter values.

        Raises:
            RuntimeError: If parameter cache is in an unexpected state.
        """
        param_list = self._optimizer.param_groups[0]["params"]

        for param in param_list:
            if not param.requires_grad:
                if param in self._param_cache:
                    raise RuntimeError("Unexpected parameter cache state")
                continue
            self._param_cache[param] = param.clone().detach()

    @jaxtyped(typechecker=typechecker)
    def _compute_inner_product(self) -> Num[Tensor, ""]:
        """
        Compute inner product term used in the Newton update.

        Returns:
            inner_product: The inner product term, as a scalar tensor.
        """
        param_list = self._optimizer.param_groups[0]["params"]

        # Accumulate inner product
        # - Here we assume `param.grad` is the "next loss" gradient
        # - This assumes that all parameters are on the same device
        device = param_list[0].device
        inner_product = torch.tensor(0.0).to(device).requires_grad_(False)

        for param in param_list:
            if not param.requires_grad:
                continue
            param_grad_flat = param.grad.flatten()
            param_update_flat = self._get_param_update(param).flatten()
            inner_product.add_(torch.inner(param_grad_flat, param_update_flat))

        return inner_product

    @jaxtyped(typechecker=typechecker)
    def _compute_next_lr(self) -> Num[Tensor, ""]:
        """
        Compute "next" learning rate.

        Returns:
            next_lr: "Next" learning rate, as a scalar tensor.
        """
        inner_product = self._compute_inner_product()
        curr_lr = self._optimizer.param_groups[0]["lr"]

        next_lr_num = (curr_lr ** 2.0) * inner_product  # fmt: skip
        next_lr_den = 2.0 * (self._curr_loss - self._next_loss - (curr_lr * inner_product))
        next_lr = curr_lr * (
            1.0 + self._config.gamma * (next_lr_num / (self._config.epsilon + next_lr_den))
        )

        return next_lr

    @jaxtyped(typechecker=typechecker)
    def _get_param_update(self, param: nn.Parameter) -> Num[Tensor, "..."]:
        param_group = self._optimizer.param_groups[0]
        curr_lr = param_group["lr"]
        param_prev = self._param_cache[param]
        param_update = param.clone().detach().sub_(param_prev).div_(-1.0 * curr_lr)
        if self._config.cache_updates:
            self._update_cache[param] = param_update
        return param_update

    @jaxtyped(typechecker=typechecker)
    def step_setup(
        self, loss: Num[Tensor, ""], x: Num[Tensor, "..."], y: Num[Tensor, "..."]
    ) -> None:
        """
        Run setup before `step()` call.

        This assumes that that `optimizer.step()` has just been called, so that
        the call to `loss_criterion(...)` computes the "next loss".

        Args:
            loss: Current loss.
            x: Input to model.
            y: Target output from model.
        """
        self._curr_loss = loss
        self._next_loss = self._config.loss_criterion(self._config.model(x), y)
        self._next_loss.backward()

    def get_lr(self) -> list[float]:
        """
        Get current learning rate.

        Returns:
            lr: List containing current learning rate.
        """
        # Skip initial step
        # TODO: Check order of operations
        if self._step_count == 1:
            return [self._optimizer.param_groups[0]["lr"]]

        # Compute "next" learning rate
        # TODO: Handle tensor LR
        lr = self._compute_next_lr()

        # Clamp "next" learning rate
        lr = torch.clamp(lr, self._config.clamp_min, self._config.clamp_max)

        # Refresh cached parameters
        self._refresh_param_cache()

        # Return updated learning rate
        return [lr]
