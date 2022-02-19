# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union

import torch
from vissl.models.heads import MLP, register_model_head


@register_model_head("skip_mlp")
class SkipMLP(MLP):
    """
    An MLP that also returns its input. Useful for constructing prediction heads
    on top of projection heads, as used in BYOL, SimSiam, NNCLR, etc.

    NOTE: The only planned use case is as a final prediction head. Can we imagine
    any use cases as an intermediate head module?
    """

    def forward(
        self, batch: Union[torch.Tensor, List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Args:
            batch: 2D torch tensor or 4D tensor of shape `N x C x 1 x 1`, or
                list of such tensors

        Returns:
            out: 2D output torch tensor
            batch: 2D input torch tensor
        """
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        out = super(SkipMLP, self).forward(batch)
        return [out, batch]
