# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pprint
from typing import List, Union

import torch
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.config import AttrDict


@register_loss("simsiam_loss")
class SimSiamLoss(ClassyLoss):
    """
    SimSiam Loss as defined in the paper https://arxiv.org/abs/2011.10566.
    """

    def __init__(self, config: AttrDict):
        super().__init__()
        self.loss_config = config

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates SimSiamLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            SimSiamLoss instance.
        """
        return cls(loss_config)

    def forward(
        self, output: Union[List[torch.Tensor], torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:

        assert isinstance(
            output, (list, tuple)
        ), f"SimSiamLoss expects a list or tuple; got {type(output)}"

        # final prediction head is a SkipMLP which returns [output, input]
        # pred is final output, proj is input i.e. projection output
        pred, proj = output

        # stop-grad
        proj = proj.detach()

        proj = nn.functional.normalize(proj, dim=1)
        pred = nn.functional.normalize(pred, dim=1)

        # split according to the two views
        proj_1, proj_2 = torch.chunk(proj, 2)
        pred_1, pred_2 = torch.chunk(pred, 2)

        loss = 0.5 * (cosine_loss(pred_1, proj_2) + cosine_loss(pred_2, proj_1))
        return loss

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)


def cosine_loss(z1, z2):
    """
    SimSiam negative cosine similarity loss.
    """
    return 1.0 - (z1 * z2).sum(dim=1).mean()
