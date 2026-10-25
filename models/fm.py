"""Second-order FM with logistic loss and a handwritten Adam optimizer."""
import numpy as np
from scipy import sparse
from scipy.special import expit


class FactorizationMachine:
    def __init__(self, factors=16, epochs=10, learning_rate=0.01,
                 regularization=0.0001, negatives=2, batch_size=2048,
                 metadata=True, seed=42):
        self.factors = factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.negatives = negatives
        self.batch_size = batch_size
        self.metadata = metadata
        self.seed = seed

    @staticmethod
    def predict_rows(x, bias, linear, factors):
        projection = x @ factors
        interaction = 0.5 * (projection ** 2 - x.multiply(x) @ (factors ** 2)).sum(axis=1)
        return bias + x @ linear + interaction

    @staticmethod
    def gradients(x, labels, bias, linear, factors, regularization):
        projection = x @ factors
        scores = FactorizationMachine.predict_rows(x, bias, linear, factors)
        error = (expit(scores) - labels) / len(labels)
        grad_bias = error.sum()
        grad_linear = np.asarray(x.T @ error) + regularization * linear
        grad_factors = np.asarray(x.T @ (error[:, None] * projection))
        grad_factors -= np.asarray(x.multiply(x).T @ error)[:, None] * factors
        grad_factors += regularization * factors
        return grad_bias, grad_linear, grad_factors

    def fit(self, train, user_features, item_features):
        rng = np.random.default_rng(self.seed)
        n_users, n_items = user_features.shape[0], item_features.shape[0]
        self.uf = sparse.eye(n_users, format='csr')
        self.itf = sparse.eye(n_items, format='csr')
        if self.metadata:
            self.uf = sparse.hstack([self.uf, user_features], format='csr')
            self.itf = sparse.hstack([self.itf, item_features], format='csr')
        users = train.user_idx.to_numpy()
        items = train.item_idx.to_numpy()
        labels = (train.rating.to_numpy() >= 4).astype(float)
        known = [set() for _ in range(n_users)]
        for user, item in zip(users, items):
            known[user].add(item)
        negative_users = np.repeat(users[labels == 1], self.negatives)
        negative_items = rng.integers(n_items, size=len(negative_users))
        for idx, user in enumerate(negative_users):
            if len(known[user]) == n_items:
                raise ValueError('No unobserved item available for negative sampling')
            while negative_items[idx] in known[user]:
                negative_items[idx] = rng.integers(n_items)
        users = np.concatenate([users, negative_users])
        items = np.concatenate([items, negative_items])
        labels = np.concatenate([labels, np.zeros(len(negative_users))])
        x = sparse.hstack([self.uf[users], self.itf[items]], format='csr')
        n_features = x.shape[1]
        self.bias = np.array(0.0)
        self.linear = np.zeros(n_features)
        self.factors_ = rng.normal(0, 0.03, (n_features, self.factors))
        params = [self.bias, self.linear, self.factors_]
        first = [np.zeros_like(p) for p in params]
        second = [np.zeros_like(p) for p in params]
        step = 0
        self.loss_history = []
        for _ in range(self.epochs):
            order = rng.permutation(len(labels))
            for start in range(0, len(order), self.batch_size):
                ids = order[start:start + self.batch_size]
                gradients = self.gradients(x[ids], labels[ids], *params, self.regularization)
                step += 1
                for p, m, v, g in zip(params, first, second, gradients):
                    m *= 0.9
                    m += 0.1 * g
                    v *= 0.999
                    v += 0.001 * g * g
                    p -= self.learning_rate * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
            # Fixed training sample: diagnostic only, never early stopping on test.
            count = min(10000, len(labels))
            logits = self.predict_rows(x[:count], *params)
            self.loss_history.append(float(np.mean(np.logaddexp(0, logits) - labels[:count] * logits)))
        self._prepare_scores()
        return self

    def _prepare_scores(self):
        split = self.uf.shape[1]
        self.user_projection = self.uf @ self.factors_[:split]
        self.item_projection = self.itf @ self.factors_[split:]
        self.user_offset = self.predict_rows(self.uf, 0, self.linear[:split], self.factors_[:split])
        self.item_offset = self.predict_rows(self.itf, 0, self.linear[split:], self.factors_[split:])

    def score(self, users):
        # Exact FM algebra for disjoint user-side and item-side feature indices.
        return (self.bias + self.user_offset[users, None] + self.item_offset[None, :]
                + self.user_projection[users] @ self.item_projection.T)
