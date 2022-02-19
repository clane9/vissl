# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn
from vissl.losses.memory_bank import MemoryBank


QUEUE_SIZE = 2048
BATCH_SIZE = 64
EMBEDDING_DIM = 128


class TestMemoryBank(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2022)
        self.queue = MemoryBank(EMBEDDING_DIM, QUEUE_SIZE)
        self.query = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        self.query = nn.functional.normalize(self.query, dim=1)

    def test_dot(self):
        sim = self.queue.dot(self.query)
        assert sim.shape == (BATCH_SIZE, QUEUE_SIZE)
        assert sim.min() >= -1 and sim.max() <= 1

    def test_neighbors(self):
        k = 5
        nbrs, sim = self.queue.neighbors(self.query, k)
        assert nbrs.shape == (BATCH_SIZE, k, EMBEDDING_DIM)
        assert sim.shape == (BATCH_SIZE, k)

        # check decreasing
        diff = sim[:, :-1] - sim[:, 1:]
        assert torch.all(diff >= 0)

        # check correct similarity
        sim2 = torch.einsum("nd,nd->n", self.query, nbrs[:, 0])
        assert torch.allclose(sim[:, 0], sim2)

    def test_nearest_neighbor(self):
        nbr = self.queue.nearest_neighbor(self.query)
        nbrs = self.queue.neighbors(self.query, 5)[0]
        assert torch.allclose(nbr, nbrs[:, 0])

    def test_random_walk(self):
        self.queue.random_walk(self.query, 4, 5, temperature=0.1, normalize=True)
        self.queue.random_walk(self.query, 4, 1, temperature=0.1, normalize=True)

    def test_dequeue_and_enqueue(self):
        self.queue.dequeue_and_enqueue(self.query)
        assert torch.allclose(self.queue.queue[:, :BATCH_SIZE], self.query.T)
        assert self.queue.queue_ptr == BATCH_SIZE
