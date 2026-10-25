"""ALS with enthusiasm relative to user habits as its preference target."""
import numpy as np
from scipy import sparse
from .ials import IALS


def make_custom_targets(train, shape, strength=0.5, prior_count=5.0):
    sums = np.bincount(train.user_idx, weights=train.rating, minlength=shape[0])
    counts = np.bincount(train.user_idx, minlength=shape[0])
    mean = (sums + prior_count * train.rating.mean()) / (counts + prior_count)
    positive = train[train.rating >= 4]
    absolute = (positive.rating.to_numpy() - 4.5) / 0.5
    relative = np.tanh(positive.rating.to_numpy() - mean[positive.user_idx])
    target = 1 + strength * (absolute + relative) / 2
    return sparse.csr_matrix(
        (target, (positive.user_idx, positive.item_idx)), shape=shape
    )


class CustomALS(IALS):
    def __init__(self, strength=0.5, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength

    def fit(self, interactions, train):
        targets = make_custom_targets(train, interactions.shape, self.strength)
        return super().fit(interactions, targets=targets)
