Train NNCLR model
===============================

VISSL reproduces the self-supervised approach called :code:`NNCLR` presented in **With a
Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual
Representations** which was proposed by **Debidatta Dwibedi, Yusuf Aytar, Jonathan
Tompson, Pierre Sermanet and Andrew Zisserman** in `this paper
<https://arxiv.org/pdf/2104.14548>`_.

How to train NNCLR model
----------------------------------

VISSL provides a yaml configuration file containing the exact hyperparameter settings to
reproduce the model. VISSL implements all the components including loss, data
augmentations, collators etc required for this approach.

To train ResNet-50 model on 8-nodes (64-gpus) on ImageNet-1K dataset:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/nnclr/nnclr_8node_resnet

Vary the training loss settings
---------------------------------

Users can adjust several settings from command line to train the model with different
hyperparams. For example: to use a different temperature 0.2 for logits and a different
queue size 32768, the training command would look like:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/nnclr/nnclr_8node_resnet \
        config.LOSS.nnclr_loss.temperature=0.2 \
        config.LOSS.nnclr_loss.queue_size=32768

The full set of loss params that VISSL allows modifying:

.. code-block:: yaml

    nnclr_loss:
      embedding_dim: 256  # output dimension for projection/prediction heads
      queue_size: 98304   # size of the nearest neighbor memory bank
      temperature: 0.1    # logit temperature

Vary the head architecture
---------------------------------

The default :code:`NNCLR` configuration uses a prediction head, represented as a :code:`skip_mlp`. To train without a prediction head, remove this module from the head params. For example:

.. code-block:: yaml

    HEAD:
      PARAMS: [
        ["mlp", {"dims": [2048, 2048, 2048], "use_relu": True, "use_bn": True, "use_bias": False, "skip_last_layer_relu_bn": False}],
        ["mlp", {"dims": [2048, 256], "use_relu": False, "use_bn": True, "use_bias": False, "skip_last_layer_relu_bn": False}],
      ]
