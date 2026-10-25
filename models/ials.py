"""Implicit weighted ALS; unseen pairs are included analytically."""
import numpy as np


class IALS:
    def __init__(self, factors=32, regularization=1.0, alpha=20.0,
                 iterations=12, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def _update(self, targets, fixed):
        gram = fixed.T @ fixed
        base = gram + self.regularization * np.eye(self.factors)
        result = np.zeros((targets.shape[0], self.factors))
        for row in range(targets.shape[0]):
            start, end = targets.indptr[row:row + 2]
            ids = targets.indices[start:end]
            if len(ids) == 0:
                continue
            values = targets.data[start:end]
            vectors = fixed[ids]
            lhs = base + self.alpha * (vectors.T @ vectors)
            rhs = (1 + self.alpha) * (vectors.T @ values)
            result[row] = np.linalg.solve(lhs, rhs)
        return result

    def fit(self, interactions, targets=None):
        targets = interactions if targets is None else targets
        targets = targets.astype(np.float64).tocsr()
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.05, (targets.shape[1], self.factors))
        for _ in range(self.iterations):
            self.user_factors = self._update(targets, self.item_factors)
            self.item_factors = self._update(targets.T.tocsr(), self.user_factors)
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
