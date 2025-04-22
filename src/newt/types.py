"""Custom types."""

import torch
from jaxtyping import Float
from torch import Tensor, nn

Optimizer = torch.optim.Optimizer
ParamTensorDict = dict[nn.Parameter, Float[Tensor, "..."]]
LossCriterion = torch.nn.modules.loss._Loss  # pylint: disable=protected-access
