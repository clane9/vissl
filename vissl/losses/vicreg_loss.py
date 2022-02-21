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

        # global mean and variance across replicas, similar to SyncBatchNorm
        # TODO: check carefully
        m1, v1 = sync_mean_var(z1)
        m2, v2 = sync_mean_var(z2)

        # invariance + variance + covariance loss
        loss = (
            self.loss_config.lambda_ * mse_loss(z1, z2)
            + self.loss_config.mu * (var_loss(v1) + var_loss(v2))
            + self.loss_config.nu * (cov_loss(z1, m1) + cov_loss(z2, m2))
        )
        return loss

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "lambda_": self.loss_config.lambda_,
            "mu": self.loss_config.mu,
            "nu": self.loss_config.nu,
        }
        return pprint.pformat(repr_dict, indent=2)


def sync_mean_var(z: torch.Tensor):
    # synchronized mean, var across replicas with grads
    # NOTE: assuming all replica batch sizes equal
    mean = z.mean(dim=0, keepdim=True)
    means = gather_from_all(mean)
    mean = means.mean(dim=0)

    batch_size = z.shape[0]
    world_size = means.shape[0]
    effective_batch_size = world_size * batch_size

    var = (z - mean).pow(2).sum(dim=0, keepdim=True)
    vars = gather_from_all(var)
    var = vars.sum(dim=0).div(effective_batch_size - 1)
    return mean, var


def mse_loss(z1: torch.Tensor, z2: torch.Tensor):
    return (z1 - z2).pow(2).sum(dim=1).mean()


def var_loss(var: torch.Tensor):
    std = torch.sqrt(var + 1e-4)
    return F.relu(1 - std).mean()


def cov_loss(z: torch.Tensor, mean: torch.Tensor):
    n, d = z.shape
    z = z - mean
    # TODO: don't think we need to sync cov since the loss is batch separable
    # but check to be sure
    cov = (z.T @ z) / (n - 1)
    cov = cov - cov.diag().diag()
    return cov.pow(2).sum().div(d)
