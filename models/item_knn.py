"""Item-based cosine neighbours with overlap shrinkage."""
import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, neighbors=50, shrinkage=10):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, interactions):
        self.history = interactions.astype(np.float64).tocsr()
        gram = (self.history.T @ self.history).tocsc()
        norms = np.sqrt(gram.diagonal())
        rows, cols, values = [], [], []
        for item in range(gram.shape[1]):
            start, end = gram.indptr[item:item + 2]
            ids = gram.indices[start:end]
            overlap = gram.data[start:end]
            scores = overlap / np.maximum(norms[ids] * norms[item], 1e-12)
            scores *= overlap / (overlap + self.shrinkage)
            valid = (ids != item) & (scores > 0)
            ids, scores = ids[valid], scores[valid]
            order = np.lexsort((ids, -scores))[:self.neighbors]
            rows.extend(ids[order])
            cols.extend([item] * len(order))
            values.extend(scores[order])
        self.weights = sparse.csr_matrix((values, (rows, cols)), shape=gram.shape)
        return self

    def score(self, users):
        return (self.history[users] @ self.weights).toarray()
