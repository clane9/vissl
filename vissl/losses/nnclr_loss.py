# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pprint
from typing import List, Union

import torch
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.config import AttrDict
from vissl.losses.memory_bank import MemoryBank
from vissl.utils.misc import concat_all_gather


@register_loss("nnclr_loss")
class NNCLRLoss(ClassyLoss):
    """
    NNCLR Loss proposed by Debidatta Dwibedi et al. in "With a Little Help from
    My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations".
    See https://arxiv.org/abs/2104.14548 for details.

    Config params:
        embedding_dim (int): head output output dimension
        queue_size (int): number of elements in queue
        temperature (float): temperature to use on the logits

    NOTE: A natural generalization might be to generate more diverse neighbors
    with random walks on the nearest neighbor graph. Although this is similar to
    the nn k > 1 variants the authors test in Table 7 (b), which seemed to have
    worse performance.
    """

    def __init__(self, config: AttrDict):
        super().__init__()
        self.loss_config = config

        # Create the queue
        self.queue = MemoryBank(
            self.loss_config.embedding_dim, self.loss_config.queue_size
        )

        self.criterion = nn.CrossEntropyLoss()
        self.initialized = False

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates NNCLRLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            NNCLRLoss instance.
        """
        return cls(loss_config)

    def forward(
        self, output: Union[List[torch.Tensor], torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            # no prediction head
            pred = proj = output
            proj = nn.functional.normalize(proj, dim=1)
        else:
            # final prediction head is a SkipMLP which returns [output, input]
            # pred is final output, proj is input i.e. projection output
            pred, proj = output
            pred = nn.functional.normalize(pred, dim=1)
            proj = nn.functional.normalize(proj, dim=1)

        if not self.initialized:
            self.queue = self.queue.to(proj.device)
            self.initialized = True

        # note, nearest neighbor query also stops gradients
        nbrs = self.queue.nearest_neighbor(proj.detach())

        # split according to the two views
        # TODO: assuming only two views. assert somewhere.
        pred_1, pred_2 = torch.chunk(pred, 2)
        nbrs_1, nbrs_2 = torch.chunk(nbrs, 2)

        # similarities between nearest neighbors and predictions
        sim_1_2 = torch.matmul(nbrs_1, pred_2) / self.loss_config.temperature
        sim_2_1 = torch.matmul(nbrs_2, pred_1) / self.loss_config.temperature

        # transpose similarities are included to symmetrize the loss
        # see Section 3.2 Implementation details, and also keras example:
        # https://github.com/keras-team/keras-io/blob/master/examples/vision/nnclr.py
        logits = torch.cat([sim_1_2, sim_1_2.T, sim_2_1, sim_2_1.T])
        labels = torch.tile(torch.arange(proj.shape[0], device=proj.device), (4,))

        # update queue using only one view, following original authors.
        proj_1, _ = torch.chunk(proj, 2)
        proj_1s = concat_all_gather(proj_1)
        self.queue.dequeue_and_enqueue(proj_1s)

        return self.criterion(logits, labels)

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)
