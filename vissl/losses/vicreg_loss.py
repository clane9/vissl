# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pprint

import torch
from classy_vision.losses import ClassyLoss, register_loss
from torch.nn import functional as F
from vissl.config import AttrDict
from vissl.utils.distributed_utils import gather_from_all


@register_loss("vicreg_loss")
class VICRegLoss(ClassyLoss):
    """
    This is the loss proposed for VICReg in https://arxiv.org/pdf/2105.04906.pdf. See
    the paper and the official code at https://github.com/facebookresearch/vicreg for
    details.

    Config params:
        sim_coeff (float): Invariance regularization loss coefficient
        std_coeff (float): Variance regularization loss coefficient
        cov_coeff (float): Covariance regularization loss coefficient
    """

    def __init__(self, config: AttrDict):
        super().__init__()
        self.loss_config = config

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates VICRegLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            VICRegLoss instance.
        """
        return cls(loss_config)

    def forward(self, output: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # copied from https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
        # see also Algorithm 1 in https://arxiv.org/pdf/2105.04906.pdf

        x, y = torch.chunk(output, 2)

        repr_loss = F.mse_loss(x, y)

        # synchronize across replicas, since var, cov loss not batch separable
        x = gather_from_all(x)
        y = gather_from_all(y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        batch_size, num_features = x.shape
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss_x = off_diagonal(cov_x).pow_(2).sum().div(num_features)
        cov_loss_y = off_diagonal(cov_y).pow_(2).sum().div(num_features)
        cov_loss = cov_loss_x + cov_loss_y

        loss = (
            self.loss_config.sim_coeff * repr_loss
            + self.loss_config.std_coeff * std_loss
            + self.loss_config.cov_coeff * cov_loss
        )
        return loss

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "sim_coeff": self.loss_config.sim_coeff,
            "std_coeff": self.loss_config.std_coeff,
            "cov_coeff": self.loss_config.cov_coeff,
        }
        return pprint.pformat(repr_dict, indent=2)


def off_diagonal(x: torch.Tensor):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
