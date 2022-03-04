Train VICReg model
===============================

VISSL reproduces the self-supervised approach called :code:`VICReg` presented in
**VICReg: Variance-invariance-covariance regularization for self-supervised learning**
which was proposed by **Adrien Bardes, Jean Ponce and Yann LeCun** in `this paper
<https://arxiv.org/pdf/2105.04906.pdf>`_.

How to train VICReg model
----------------------------------

VISSL provides a yaml configuration file containing the exact hyperparameter settings to
reproduce the model. VISSL implements all the components including loss, data
augmentations, collators etc required for this approach.

To train ResNet-50 model on 4-nodes (32-gpus) on ImageNet-1K dataset:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/vicreg/vicreg_4node_resnet

Vary the training loss settings
---------------------------------

Users can adjust several settings from command line to train the model with different
hyperparams. For example: to use weaker invariance and variance regularization
coefficients, the training command would look like:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/vicreg/vicreg_4node_resnet \
        config.LOSS.vicreg_loss.sim_coeff=10.0 \
        config.LOSS.vicreg_loss.std_coeff=10.0

The full set of loss params that VISSL allows modifying:

.. code-block:: yaml

    vicreg_loss:
      sim_coeff: 25.0  # invariance regularization coefficient
      std_coeff: 25.0  # variance regularization coefficient
      cov_coeff: 1.0   # covariance regularization coefficient
