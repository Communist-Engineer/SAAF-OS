import json
import os
import numpy as np
from typing import Tuple
from modules.planner_utils import vectorize_plan


def load_plan_dataset(path: str = "memory/episodes.jsonl") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load episodes from JSONL and return arrays:
    X: latent states z_t
    Y: vectorized plan features
    """
    if not os.path.exists(path):
        return np.empty((0, )), np.empty((0, ))
    X_list = []
    Y_list = []
    with open(path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                z_t = np.array(record.get('z_t', []), dtype=float)
                plan = record.get('plan', {})
                plan_vec = vectorize_plan(plan)
                if z_t.size > 0 and plan_vec.size > 0:
                    X_list.append(z_t)
                    Y_list.append(plan_vec)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    if not X_list:
        return np.empty((0, )), np.empty((0, ))
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y