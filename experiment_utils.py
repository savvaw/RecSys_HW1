"""Hyperparameter search and final model comparison."""
import pandas as pd
from models import ItemKNN, IALS, EASE, FactorizationMachine, CustomALS
from evaluation import evaluate

GRIDS = {
    'ItemKNN': [{'neighbors': k, 'shrinkage': s} for k in [30, 100] for s in [0, 20]],
    'iALS': [{'factors': f, 'regularization': r, 'alpha': 20.0, 'iterations': 10, 'seed': 42}
             for f in [16, 32] for r in [1.0, 10.0]],
    'EASE': [{'regularization': r} for r in [50.0, 200.0]],
    'FM': [{'factors': 16, 'epochs': e, 'learning_rate': 0.01,
            'regularization': 0.0001, 'negatives': 2, 'batch_size': 2048,
            'metadata': True, 'seed': 42} for e in [8, 16]],
}
CUSTOM_STRENGTHS = [0.25, 0.75]


def fit_model(name, params, stage):
    constructors = {'ItemKNN': ItemKNN, 'iALS': IALS, 'EASE': EASE,
                    'FM': FactorizationMachine, 'CustomALS': CustomALS,
                    'FM_ID_only': FactorizationMachine}
    model = constructors[name](**params)
    if name.startswith('FM'):
        return model.fit(stage['train'], stage['user_features'], stage['item_features'])
    if name == 'CustomALS':
        return model.fit(stage['interactions'], stage['train'])
    return model.fit(stage['interactions'])


def tune(stage):
    rows, selected = [], {}
    for name in ['ItemKNN', 'iALS', 'EASE', 'FM', 'CustomALS']:
        if name == 'CustomALS':
            grid = [dict(selected['iALS'], strength=s) for s in CUSTOM_STRENGTHS]
        else:
            grid = GRIDS[name]
        best_score = -1
        for params in grid:
            print(f'Validation: {name} {params}', flush=True)
            model = fit_model(name, params, stage)
            metrics = evaluate(model, stage)
            rows.append({'model': name, 'params': params, **metrics})
            if metrics['NDCG@10'] > best_score:
                best_score = metrics['NDCG@10']
                selected[name] = params.copy()
            del model
    selected['FM_ID_only'] = dict(selected['FM'], metadata=False)
    return pd.DataFrame(rows), selected


def final_comparison(stage, selected):
    rows = []
    for name, params in selected.items():
        print(f'Test: {name}', flush=True)
        model = fit_model(name, params, stage)
        rows.append({'model': name, 'params': params, **evaluate(model, stage)})
        del model
    return pd.DataFrame(rows)
