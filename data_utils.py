"""Download, chronological split and historical metadata features."""
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

SEED = 42
DOWNLOAD_URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Digital_Music_5.json.gz'
TRAIN_END = pd.Timestamp('2015-01-01', tz='UTC')
VALIDATION_END = pd.Timestamp('2016-01-01', tz='UTC')
USER_FEATURES = ['log_activity', 'verified_share']
ITEM_FEATURES = ['log_popularity', 'log_age_days', 'verified_share']


def download_data(directory='work/raw'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'Digital_Music_5.json.gz'
    if not path.exists():
        temporary = path.with_suffix('.part')
        urllib.request.urlretrieve(DOWNLOAD_URL, temporary)
        temporary.replace(path)
    return path


def load_data(path):
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            rows.append({
                'user': record.get('reviewerID'), 'item': record.get('asin'),
                'rating': record.get('overall'), 'timestamp': record.get('unixReviewTime'),
                'verified': record.get('verified', np.nan),
            })
    raw = pd.DataFrame(rows)
    report = {'raw_rows': len(raw), 'raw_users': raw.user.nunique(),
              'raw_items': raw.item.nunique(), 'missing': raw.isna().sum().to_dict(),
              'duplicate_used_fields': int(raw.duplicated().sum())}
    clean = raw.dropna(subset=['user', 'item', 'rating', 'timestamp']).copy()
    clean['rating'] = pd.to_numeric(clean.rating, errors='coerce')
    clean['timestamp'] = pd.to_datetime(clean.timestamp, unit='s', utc=True, errors='coerce')
    clean = clean[clean.rating.between(1, 5) & clean.timestamp.notna()]
    report['invalid_or_missing_rows'] = len(raw) - len(clean)
    clean = clean.sort_values('timestamp', kind='stable')
    report['repeated_user_item_rows'] = int(clean.duplicated(['user', 'item']).sum())

    clean = clean.drop_duplicates(['user', 'item'], keep='first').reset_index(drop=True)
    clean['verified'] = clean.verified.fillna(False).astype(float)
    report.update(clean_rows=len(clean), clean_users=clean.user.nunique(),
                  clean_items=clean.item.nunique(),
                  sparsity=1 - len(clean) / (clean.user.nunique() * clean.item.nunique()))

    return raw, clean, report


def split_data(clean):
    train = clean[clean.timestamp < TRAIN_END].copy()
    validation = clean[(clean.timestamp >= TRAIN_END) & (clean.timestamp < VALIDATION_END)].copy()
    test = clean[clean.timestamp >= VALIDATION_END].copy()
    return train, validation, test


def standardize(values):
    values = np.asarray(values, dtype=float)
    return (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-8)


def build_stage(history, future, cutoff):
    users = sorted(history.user.unique())
    items = sorted(history.item.unique())
    umap = {u: i for i, u in enumerate(users)}
    imap = {x: i for i, x in enumerate(items)}
    train = history.copy()
    train['user_idx'] = train.user.map(umap).astype(int)
    train['item_idx'] = train.item.map(imap).astype(int)
    shape = (len(users), len(items))
    known = sparse.csr_matrix((np.ones(len(train)), (train.user_idx, train.item_idx)), shape=shape)
    positive = train[train.rating >= 4]
    interactions = sparse.csr_matrix((np.ones(len(positive)), (positive.user_idx, positive.item_idx)), shape=shape)
    user_groups = train.groupby('user_idx')
    item_groups = train.groupby('item_idx')
    user_values = np.column_stack([
        np.log1p(user_groups.size().reindex(range(shape[0])).to_numpy()),
        user_groups.verified.mean().reindex(range(shape[0])).to_numpy(),
    ])
    first_time = item_groups.timestamp.min().reindex(range(shape[1]))
    item_values = np.column_stack([
        np.log1p(item_groups.size().reindex(range(shape[1])).to_numpy()),
        np.log1p((cutoff - first_time).dt.total_seconds().to_numpy() / 86400),
        item_groups.verified.mean().reindex(range(shape[1])).to_numpy(),
    ])
    relevant = future[future.rating >= 4].copy()
    warm_user = relevant.user.isin(umap)
    warm_item = relevant.item.isin(imap)
    diagnostics = {
        'history_rows': len(train), 'positive_history': len(positive),
        'users': shape[0], 'catalogue_items': shape[1], 'future_rows': len(future),
        'future_positive_rows': len(relevant),
        'cold_user_positive_rows': int((~warm_user).sum()),
        'cold_item_positive_rows': int((~warm_item).sum()),
        'cold_any_positive_rows': int((~(warm_user & warm_item)).sum()),
    }
    relevant = relevant[warm_user & warm_item].copy()
    relevant['user_idx'] = relevant.user.map(umap).astype(int)
    relevant['item_idx'] = relevant.item.map(imap).astype(int)
    active = np.diff(interactions.indptr) >= 3
    enough_candidates = shape[1] - np.diff(known.indptr) >= 10
    eligible = active & enough_candidates
    diagnostics['inactive_positive_rows'] = int((~eligible[relevant.user_idx]).sum())
    relevant = relevant[eligible[relevant.user_idx]]
    truth = {int(u): set(group.item_idx) for u, group in relevant.groupby('user_idx')}
    diagnostics['evaluated_users'] = len(truth)
    diagnostics['evaluated_positive_rows'] = sum(map(len, truth.values()))
    return {
        'train': train, 'interactions': interactions, 'known': known, 'truth': truth,
        'user_features': sparse.csr_matrix(standardize(user_values)),
        'item_features': sparse.csr_matrix(standardize(item_values)),
        'raw_user_features': pd.DataFrame(user_values, columns=USER_FEATURES),
        'raw_item_features': pd.DataFrame(item_values, columns=ITEM_FEATURES),
        'diagnostics': diagnostics,
    }
