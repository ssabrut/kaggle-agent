"""Shared CV folds. Written once at init so every candidate script is scored on the same split."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from kagent.config import CompetitionConfig
from kagent.workspace import Workspace

CLASSIFICATION_METRICS = {"accuracy", "auc", "f1_macro", "logloss"}


def make_folds(ws: Workspace, config: CompetitionConfig, n_splits: int = 5, seed: int = 0) -> bool:
    """Write data/public/folds.csv (id, fold). Returns False if there is no train.csv yet."""
    train_path = ws.public / "train.csv"
    if not train_path.exists():
        return False
    target = ws.target_column(config)
    train = pd.read_csv(train_path, usecols=[config.id_column, target])

    if config.metric in CLASSIFICATION_METRICS:
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(train, train[target])
    else:
        splits = KFold(n_splits, shuffle=True, random_state=seed).split(train)

    fold = np.empty(len(train), dtype=int)
    for k, (_, valid_idx) in enumerate(splits):
        fold[valid_idx] = k
    pd.DataFrame({config.id_column: train[config.id_column], "fold": fold}).to_csv(
        ws.public / "folds.csv", index=False
    )
    return True
