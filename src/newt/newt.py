"""A Newton-like learning rate scheduler."""

from dataclasses import dataclass
from typing import Union

import torch
from jaxtyping import Num, jaxtyped
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from typeguard import typechecked as typechecker

from .types import LossCriterion, Optimizer, ParamTensorDict

_HESSIAN_NOT_PSD_FACTOR = 1.0
_LR_FACTOR_CLAMP_MIN = 0.0
_LR_FACTOR_CLAMP_MAX = 1.05
_SCALEBACK_FACTOR = 0.1


@dataclass
class NewtConfig:
    """
    Dataclass to hold `Newt` configuration.

    Attrs:
        model: Model instance (default = None).
        loss_criterion: Loss criterion instance (default = None).
        use_alternate_approx: Use alternate approximation of the Hessian
            product (default = False).  The standard approximation of the
            Hessian product is the one used in [1].
        gamma: Slow-adaptation parameter for LR updates (default = 1e-3).        
        epsilon: Constant for divide-by-zero protection (default = 1e-9).
        lr_clamp_min: Learning rate lower bound (default = 1e-7).
        lr_clamp_max: Learning rate upper bound (default = 1.0).
        cache_updates: Keep a cache of parameter updates (default = False).
            This is only useful for development and testing purposes.
    """

    model: nn.Module
    loss_criterion: LossCriterion
    use_alternate_approx: bool = False
    gamma: float = 1e-3
    epsilon: float = 1e-9
    lr_clamp_min: float = 1e-7
    lr_clamp_max: float = 1.0
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

        self._curr_loss: Tensor = torch.tensor(0.0)
        self._curr_inner_product: Tensor = torch.tensor(0.0)
        self._lookahead_loss: Tensor = torch.tensor(0.0)
        self._lookahead_inner_product: Tensor = torch.tensor(0.0)
        self._param_cache: ParamTensorDict = ParamTensorDict()
        self._update_cache: ParamTensorDict = ParamTensorDict()

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

        This is used to compute either the current inner product or lookahead
        inner product, depending on the gradient state.

        Returns:
            inner_product: The inner product term, as a scalar tensor.
        """
        param_list = self._optimizer.param_groups[0]["params"]

        # Accumulate inner product
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
    def _compute_lr(self) -> Num[Tensor, ""]:
        """
        Compute updated learning rate.

        Returns:
            next_lr: Updated learning rate, as a scalar tensor.
        """
        curr_lr: Union[float, Tensor] = self._optimizer.param_groups[0]["lr"]

        if not self._config.use_alternate_approx:
            next_lr_num = self._lookahead_inner_product
            next_lr_den = self._curr_inner_product - self._lookahead_inner_product
        else:
            next_lr_num = curr_lr * self._lookahead_inner_product
            next_lr_den = 2.0 * (self._curr_loss - self._lookahead_loss - next_lr_num)

        # Heuristics to handle special cases:
        # - Increasing loss: Drastically scale back multiplicative factor
        # - Negative denominator: Hessian is not positive-semidefinite, skip
        # - General case: Multiplicative factor influence is capped above
        if self._lookahead_loss > self._curr_loss:
            next_lr_factor = torch.tensor(_SCALEBACK_FACTOR)
        elif next_lr_den < self._config.epsilon:
            next_lr_factor = torch.tensor(_HESSIAN_NOT_PSD_FACTOR)
        else:
            next_lr_den = self._config.epsilon + next_lr_den
            next_lr_factor = 1.0 + self._config.gamma * (next_lr_num / next_lr_den)
            next_lr_factor = torch.clamp(next_lr_factor, _LR_FACTOR_CLAMP_MIN, _LR_FACTOR_CLAMP_MAX)

        next_lr = next_lr_factor * curr_lr
        next_lr = torch.clamp(next_lr, self._config.lr_clamp_min, self._config.lr_clamp_max)
        
        return next_lr

    @jaxtyped(typechecker=typechecker)
    def _get_param_update(self, param: nn.Parameter) -> Num[Tensor, "..."]:
        curr_lr = self._optimizer.param_groups[0]["lr"]
        
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
        the call to `loss_criterion(...)` computes the lookahead loss.

        Args:
            loss: Current loss.
            x: Input to model.
            y: Target output from model.
        """
        # Alternate approximation uses current loss and current inner product
        if not self._config.use_alternate_approx:
            self._curr_loss = loss
            self._curr_inner_product = self._compute_inner_product()

        # Both approximations use lookahead inner product
        self._lookahead_loss = self._config.loss_criterion(self._config.model(x), y)
        self._optimizer.zero_grad()
        self._lookahead_loss.backward()
        self._lookahead_inner_product = self._compute_inner_product()

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

        # Update learning rate
        lr = self._compute_lr()

        # Refresh cached parameters
        self._refresh_param_cache()

        # Return updated learning rate
        return [lr]
