"""Common top-10 recommendations and ranking metrics."""
import numpy as np


def recommend(model, users, known, k=10):
    scores = np.asarray(model.score(users), dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Model produced non-finite scores')
    for row, user in enumerate(users):
        scores[row, known[user].indices] = -np.inf
    return np.argsort(-scores, axis=1, kind='stable')[:, :k]


def metrics_at_k(recommendations, relevant, k=10):
    hits = np.array([item in relevant for item in recommendations[:k]], dtype=float)
    discount = 1 / np.log2(np.arange(2, k + 2))
    ideal = discount[:min(k, len(relevant))].sum()
    return float(hits @ discount / ideal), float(hits.sum() / len(relevant))


def evaluate(model, stage, k=10, batch_size=64):
    users = np.array(sorted(stage['truth']))
    if len(users) == 0:
        raise ValueError('Empty evaluation cohort')
    values = []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        recs = recommend(model, batch, stage['known'], k)
        for user, items in zip(batch, recs):
            values.append(metrics_at_k(items, stage['truth'][user], k))
    ndcg, recall = np.mean(values, axis=0)
    return {'NDCG@10': float(ndcg), 'Recall@10': float(recall)}
