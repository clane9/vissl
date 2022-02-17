# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch import Tensor, nn


class MemoryBank(nn.Module):
    """
    A memory bank i.e. queue for storing and querying recent embedded
    representations. Supports all-to-all comparison and nearest neighbor queries.
    Also introduces random walk queries, which are a natural but untested
    generalization of nearest neighbor queries. Used in SSL methods like MoCo
    and NNCLR.
    """

    def __init__(self, embedding_dim: int, queue_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size

        # TODO: Random initialization is ok? Probably. At the beginning of
        # training the embeddings are random anyway.
        self.register_buffer("queue", torch.randn(queue_size, embedding_dim))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, key: Tensor, normalize: bool = False):
        """
        Discard the oldest key from the queue, save the newest one,
        through a round-robin mechanism
        """
        batch_size = key.shape[0]
        if normalize:
            key = nn.functional.normalize(key, dim=1)

        # for simplicity, removes the case where the batch overlaps with the end
        # of the queue
        assert self.queue_size % batch_size == 0, (
            f"The queue size needs to be a multiple of the batch size. "
            f"Effective batch size: {batch_size}. Queue size:"
            f" {self.queue_size}."
        )

        # replace the keys at ptr (dequeue and enqueue)
        ptr = int(self.queue_ptr)
        self.queue[ptr : ptr + batch_size, :] = key
        ptr = (ptr + batch_size) % self.queue_size  # move pointer, round robin

        self.queue_ptr[0] = ptr

    @staticmethod
    def similarity(
        query: Tensor, key: Tensor, normalize: bool = False, diag: bool = False
    ) -> Tensor:
        """
        Compute cosine similarity between query `(n, d)` and key `(k, d)`.
        Returns a matrix of similarities `(n, k)`, or a vector `(n,)` if `diag`.
        """

        # TODO: This is maybe broadly useful. Decide whether to put it elsewhere.
        if normalize:
            query = nn.functional.normalize(query, dim=1)
            key = nn.functional.normalize(key, dim=1)

        if diag:
            sim = torch.einsum("nc,nc->n", [query, key])
        else:
            sim = torch.einsum("nc,kc->nk", [query, key])
        return sim

    def query(self, query: Tensor, normalize: bool = False) -> Tensor:
        """
        Pairwise comparison between query `(n, d)` and the full memory bank.
        Returns a matrix if similarities `(n, K)` where `n` and `K` are the
        number of query and memory bank vectors.
        """
        if normalize:
            query = nn.functional.normalize(query, dim=1)
        return self.similarity(query, self.queue)

    @torch.no_grad()
    def neighbors(
        self, query: Tensor, k: int, normalize: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Find nearest `k` neigbors for a query `(n, d)`. Returns the neighbors
        `(n, k, d)` and the similarities `(n, k)`.
        """
        sim = self.query(query, normalize=normalize)
        # (n, k)
        values, indices = torch.topk(sim, k, dim=1)
        # (n, k, d)
        neighbors = self.queue[indices, :]
        return neighbors, values

    @torch.no_grad()
    def nearest_neighbor(self, query: Tensor, normalize: bool = False) -> Tensor:
        """
        Find the nearest neighbor for each vector in the query.
        """
        neighbors, _ = self.neighbors(query, 1, normalize=normalize)
        return neighbors[:, 0, :]

    @torch.no_grad()
    def random_walk(
        self,
        query: Tensor,
        steps: int,
        k: int,
        temperature: float = 1.0,
        normalize: bool = False,
    ) -> Tensor:
        """
        Take a random walk on the `k`-nearest neighbor graph starting from the
        query vectors. `temperature` controls the softmax transition
        probabilities.
        """

        if normalize:
            query = nn.functional.normalize(query, dim=1)

        for _ in range(steps):
            neighbors, sim = self.neighbors(query, k)
            if k == 1:
                query = neighbors[:, 0, :]
                # need at least two neighbors on later steps to prevent cycles
                k = k + 1
                continue

            # prevent cyclic transitions
            identity_mask = sim >= 1.0 - 1e-6
            sim[identity_mask] = -1e6
            prob = torch.softmax(sim / temperature, dim=1)
            indices = torch.multinomial(prob, 1).squeeze(1)
            query = neighbors[torch.arange(query.size(0)), indices]
        return query
