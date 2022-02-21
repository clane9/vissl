# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pprint

import torch
from classy_vision.losses import ClassyLoss, register_loss
from torch.nn import functional as F
from vissl.config import AttrDict


@register_loss("vicreg_loss")
class VICRegLoss(ClassyLoss):
    """
    VICReg Loss.
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
        # see Algorithm 1 in https://arxiv.org/pdf/2105.04906.pdf
        # copied verbatim
        z1, z2 = torch.chunk(output, 2)

        # invariance + variance + covariance loss
        # TODO: var loss is not strictly separable across the batch, due to the
        # sqrt. Also the batch mean used to center should probably me shared.
        # Think/check what to do in distributed setting.
        # Should we all reduce the batch mean and variance to get it correct?
        loss = (
            self.loss_config.lambd * mse_loss(z1, z2)
            + self.loss_config.mu * (var_loss(z1) + var_loss(z2))
            + self.loss_config.nu * (cov_loss(z1) + cov_loss(z2))
        )
        return loss

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "lambd": self.loss_config.lambd,
            "mu": self.loss_config.mu,
            "nu": self.loss_config.nu,
        }
        return pprint.pformat(repr_dict, indent=2)


def mse_loss(z1, z2):
    return (z1 - z2).pow(2).sum(dim=1).mean()


def var_loss(z: torch.Tensor):
    # NOTE: not batch separable. It it supposed to be modified in some way?
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(1 - std).mean()


def cov_loss(z: torch.Tensor):
    n, d = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (n - 1)
    cov = cov - cov.diag().diag()
    return cov.pow(2).sum().div(d)
