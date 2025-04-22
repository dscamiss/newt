"""A Newton-like learning rate scheduler."""

import torch
from jaxtyping import Num, jaxtyped
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from typeguard import typechecked as typechecker

from newt.types import LossCriterion, Optimizer, ParamTensorDict

_DEFAULT_GAMMA = 1e-2
_DEFAULT_EPSILON = 1e-5


class Newt(LRScheduler):
    """
    The `newt` learning rate scheduler.

    Args:
        optimizer: Optimizer instance to wrap.
        last_epoch: Index of the last epoch (default = -1).
    """

    # TODO: Remove single parameter group limitation.
    def __init__(
        self,
        optimizer: Optimizer,
        loss_criterion: LossCriterion,
        gamma: float = _DEFAULT_GAMMA,
        epsilon: float = _DEFAULT_EPSILON,
        cache_updates: bool = False,
        last_epoch: int = -1,
    ) -> None:
        # Sanity check on number of parameter groups
        if len(optimizer.param_groups) != 1:
            raise ValueError("Optimizer must have exactly one parameter group")

        # Sanity check on differentiable optimizer
        if optimizer.param_groups[0]["differentiable"]:
            raise ValueError("Differentiable optimizers are not supported")

        self._optimizer = optimizer
        self._loss_criterion = loss_criterion
        self._gamma = gamma
        self._epsilon = epsilon
        self._cache_updates = cache_updates

        self._curr_loss = None
        self._next_loss = None
        self._param_cache = ParamTensorDict()
        self._update_cache = ParamTensorDict()

        # Refresh cached parameters
        self._refresh_param_cache()

        super().__init__(optimizer, last_epoch)

    def _refresh_param_cache(self) -> None:
        """
        Refresh cached parameters with current values.

        Raises:
            RuntimeError: If parameter cache is in an unexpected state.
        """
        param_group = self._optimizer.param_groups[0]
        param_list = param_group["params"]
        for param in param_list:
            if not param.requires_grad:
                if param in self._param_cache:
                    raise RuntimeError("Unexpected parameter cache state")
                continue
            self._param_cache[param] = param.clone().detach()

    @jaxtyped(typechecker=typechecker)
    def _compute_inner_product(self) -> Num[Tensor, ""]:
        """
        Compute inner product term for the Newton update.

        Returns:
            inner_product: The inner product term as a scalar tensor.
        """
        param_list = self._optimizer.param_groups[0]["params"]
        inner_product = torch.as_tensor(0.0).requires_grad_(False)

        # Accumulate inner product
        for param in param_list:
            if not param.requires_grad:
                continue
            param_update = self._get_param_update(param)
            # Here we assume `param.grad` is the "next loss" gradient
            inner_product.add_(torch.inner(param.grad.flatten(), param_update.flatten()))

        return inner_product

    @jaxtyped(typechecker=typechecker)
    def _compute_lr(self) -> Num[Tensor, ""]:
        """
        Compute next learning rate.

        Returns:
            next_lr: Next learning rate as a scalar tensor.
        """
        inner_product = self._compute_inner_product()
        curr_lr = self._optimizer.param_groups[0]["lr"]
        next_lr_num = (curr_lr ** 2.0) * inner_product  # fmt: skip
        next_lr_den = 2.0 * (self._curr_loss - self._next_loss - (curr_lr * inner_product))
        return curr_lr * (1.0 + self._gamma * (next_lr_num / (self._epsilon + next_lr_den)))

    @jaxtyped(typechecker=typechecker)
    def _get_param_update(self, param: nn.Parameter) -> Num[Tensor, "..."]:
        param_group = self._optimizer.param_groups[0]
        curr_lr = param_group["lr"]
        param_prev = self._param_cache[param]
        param_update = param.clone().detach().sub_(param_prev).div_(-1.0 * curr_lr)
        if self._cache_updates:
            self._update_cache[param] = param_update
        return param_update

    @jaxtyped(typechecker=typechecker)
    def step_setup(
        self, loss: Num[Tensor, ""], y_hat: Num[Tensor, "..."], y: Num[Tensor, "..."]
    ) -> None:
        """
        Run setup before `step()` call.

        Args:
            loss: Current loss.
            y_hat: Output.
            y: Target output.
        """
        self._curr_loss = loss
        self._next_loss = self._loss_criterion(y_hat, y)
        self._next_loss.backward()

    def get_lr(self) -> list[float]:
        """
        Get current learning rate.

        Returns:
            List containing current learning rate.
        """
        # Skip initial step
        # TODO: Check order of operations
        if self._step_count == 1:
            return [self._optimizer.param_groups[0]["lr"]]

        # Update learning rate
        # TODO: Handle tensor LR
        lr = self._compute_lr()

        # Refresh cached parameters
        self._refresh_param_cache()

        # Return updated learning rate
        return [lr]
