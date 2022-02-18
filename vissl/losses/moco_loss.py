# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import pprint
from collections import namedtuple

import torch
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.losses.memory_bank import MemoryBank
from vissl.utils.misc import concat_all_gather


_MoCoLossConfig = namedtuple(
    "_MoCoLossConfig", ["embedding_dim", "queue_size", "momentum", "temperature"]
)


class MoCoLossConfig(_MoCoLossConfig):
    """Settings for the MoCo loss"""

    @staticmethod
    def defaults() -> "MoCoLossConfig":
        return MoCoLossConfig(
            embedding_dim=128, queue_size=65536, momentum=0.999, temperature=0.2
        )


@register_loss("moco_loss")
class MoCoLoss(ClassyLoss):
    """
    This is the loss which was proposed in the "Momentum Contrast
    for Unsupervised Visual Representation Learning" paper, from Kaiming He et al.
    See http://arxiv.org/abs/1911.05722 for details
    and https://github.com/facebookresearch/moco for a reference implementation, reused here

    Config params:
        embedding_dim (int): head output output dimension
        queue_size (int): number of elements in queue
        momentum (float): encoder momentum value for the update
        temperature (float): temperature to use on the logits
    """

    def __init__(self, config: MoCoLossConfig):
        super().__init__()
        self.loss_config = config

        # Create the queue
        self.queue = MemoryBank(
            self.loss_config.embedding_dim, self.loss_config.queue_size
        )

        self.criterion = nn.CrossEntropyLoss()
        self.initialized = False

        self.key = None
        self.sample = None
        self.moco_encoder = None

        self.checkpoint = None

    @classmethod
    def from_config(cls, config: MoCoLossConfig):
        """
        Instantiates MoCoLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            MoCoLoss instance.
        """
        return cls(config)

    def forward(self, query: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Given the encoder queries, the key and the queue of the previous queries,
        compute the cross entropy loss for this batch

        Args:
            query: output of the encoder given the current batch

        Returns:
            loss
        """

        if not self.initialized:
            self.queue = self.queue.to(query.device)
            self.initialized = True

        # --
        # Normalize the encoder raw outputs
        query = nn.functional.normalize(query, dim=1)

        # --
        # Compute all the logits and the expected labels
        # Einstein sum is used in MoCo, deemed more intuitive.
        # Another option is `torch.diag(torch.matmul(query, self.key.T))`

        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [query, self.key]).unsqueeze(-1)

        # negative logits: NxK
        l_neg = self.queue.dot(query)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.loss_config.temperature

        # labels: positives are the first rank.
        # This is essentially a classification problem alongside all the samples
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(query.device)

        # ---
        # Update the queue for the next time
        # gather keys before updating queue /!\ the queue is duplicated on all GPUs
        keys = concat_all_gather(self.key)
        self.queue.dequeue_and_enqueue(keys)

        # ---
        # Then just apply the XELoss
        return self.criterion(logits, labels)

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Restore the loss state given a checkpoint

        Args:
            state_dict (serialized via torch.save)
        """

        # If the encoder has been allocated, use the normal pytorch restoration
        if self.moco_encoder is None:
            self.checkpoint = state_dict
            logging.info("Storing the checkpoint for later use")
        else:
            logging.info("Restoring checkpoint")
            super().load_state_dict(state_dict, *args, **kwargs)
