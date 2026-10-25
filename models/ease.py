"""EASE: constrained ridge regression using its analytical solution."""
import numpy as np


class EASE:
    def __init__(self, regularization=100):
        self.regularization = regularization

    def fit(self, interactions):
        self.history = interactions.astype(np.float64).tocsr()
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::gram.shape[0] + 1] += self.regularization
        precision = np.linalg.inv(gram)
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, users):
        return self.history[users] @ self.weights
