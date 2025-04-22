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

    @jaxtyped(typechecker=typechecker)
    def _get_param_update(self, param: nn.Parameter) -> Num[Tensor, "..."]:
        param_group = self._optimizer.param_groups[0]
        curr_lr = param_group["lr"]
        param_prev = self._param_cache[param]
        param_update = param.clone().detach().sub_(param_prev).div_(-1.0 * curr_lr)
        if self._cache_updates:
            self._update_cache[param] = param_update
        return param_update

    def get_lr(self) -> list[float]:
        """
        Get current learning rate.

        Returns:
            List containing current learning rate.
        """
        param_group = self._optimizer.param_groups[0]
        param_list = param_group["params"]
        curr_lr = param_group["lr"]

        # Skip initial step
        if self._step_count == 1:
            return [param_group["lr"]]

        # Compute inner product term
        #     <\nabla L(theta_t - alpha_t omega_t), omega_t>
        inner_product = torch.as_tensor(0.0).requires_grad_(False)
        for param in param_list:
            if not param.requires_grad:
                continue
            param_update = self._get_param_update(param)
            # Here, `param.grad` is the "next loss" gradient
            inner_product.add_(torch.inner(param.grad.flatten(), param_update.flatten()))

        # Refresh cached parameters
        self._refresh_param_cache()

        # Compute new learning rate
        # TODO: Handle tensor LR
        next_lr_num = (curr_lr ** 2.0) * inner_product  # fmt: skip
        next_lr_den = 2.0 * (self._curr_loss - self._next_loss - (curr_lr * inner_product))
        next_lr = curr_lr * (1.0 + self._gamma * (next_lr_num / (self._epsilon + next_lr_den)))

        # Return new learning rate
        return [next_lr]
