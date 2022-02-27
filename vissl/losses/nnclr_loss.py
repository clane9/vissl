# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pprint
from typing import List, Union

import torch
from classy_vision.generic.distributed_util import get_rank
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.config import AttrDict
from vissl.losses.memory_bank import MemoryBank
from vissl.utils.distributed_utils import gather_from_all
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

        self.dist_rank = get_rank()
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

        has_pred = isinstance(output, (list, tuple))
        if has_pred:
            # final prediction head is a SkipMLP which returns [output, input]
            # pred is final output, proj is input i.e. projection output
            pred, proj = output
        else:
            # no prediction head
            pred = proj = output

        orig_images = proj.shape[0] // 2
        device = proj.device

        if not self.initialized:
            self.queue = self.queue.to(device)
            self.initialized = True

        proj = nn.functional.normalize(proj, dim=1)
        if has_pred:
            pred = nn.functional.normalize(pred, dim=1)

        # note, nearest neighbor query also stops gradients
        nbrs = self.queue.nearest_neighbor(proj.detach())

        # split according to the two views
        pred_1, pred_2 = torch.chunk(pred, 2)
        nbrs_1, nbrs_2 = torch.chunk(nbrs, 2)

        # gather all outputs, keeping grads for predicitons
        # note all outputs must be gathered, since each term is used as negative examples
        # TODO: should these communication ops be grouped for less overhead?
        pred_1s = gather_from_all(pred_1)
        pred_2s = gather_from_all(pred_2)
        nbrs_1s = concat_all_gather(nbrs_1)
        nbrs_2s = concat_all_gather(nbrs_2)

        # each replica computes the loss for its assigned batch
        rows = slice(self.dist_rank * orig_images, (self.dist_rank + 1) * orig_images)
        labels = torch.arange(
            self.dist_rank * orig_images,
            (self.dist_rank + 1) * orig_images,
            device=device,
        )

        # similarities between nearest neighbors and predictions
        # transpose similarities are included to symmetrize the loss
        # see Section 3.2 Implementation details, and also keras example:
        # https://github.com/keras-team/keras-io/blob/master/examples/vision/nnclr.py
        sim_1_1_2 = torch.matmul(nbrs_1s[rows], pred_2s.T)
        sim_1_2_1 = torch.matmul(nbrs_2s[rows], pred_1s.T)
        sim_2_1_2 = torch.matmul(pred_1s[rows], nbrs_2s.T)
        sim_2_2_1 = torch.matmul(pred_2s[rows], nbrs_1s.T)

        logits = torch.cat([sim_1_1_2, sim_1_2_1, sim_2_1_2, sim_2_2_1]).div(
            self.loss_config.temperature
        )
        labels = torch.tile(labels, (4,))
        loss = self.criterion(logits, labels)

        # update queue using only one view, following original authors.
        # gather keys before updating queue /!\ the queue is duplicated on all GPUs
        if has_pred:
            proj_1, _ = torch.chunk(proj.detach(), 2)
            proj_1s = concat_all_gather(proj_1)
            self.queue.dequeue_and_enqueue(proj_1s)
        else:
            # pred is proj
            self.queue.dequeue_and_enqueue(pred_1s)

        return loss

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "embedding_dim": self.loss_config.embedding_dim,
            "queue_size": self.loss_config.queue_size,
            "temperature": self.loss_config.temperature,
        }
        return pprint.pformat(repr_dict, indent=2)
