import json
import os
import numpy as np
from typing import List, Dict, Any

def retrieve_similar_episode(
    z_t: np.ndarray, top_k: int = 3,
    episodes_path: str = "memory/episodes.jsonl"
) -> List[Dict[str, Any]]:
    """
    Return the top_k most similar episodes to the given latent state z_t,
    based on Euclidean distance between z_t and stored episodes' z_t vectors.
    Each returned dict includes scenario, pre_score, post_score, energy_initial,
    energy_final, and rsi_accepted.
    """
    if not os.path.exists(episodes_path):
        return []
    candidates: List[Dict[str, Any]] = []
    with open(episodes_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                z_record = np.array(record.get('z_t', []), dtype=float)
                if z_record.shape != z_t.shape:
                    continue
                dist = np.linalg.norm(z_record - z_t)
                candidates.append({
                    'distance': dist,
                    'scenario': record.get('scenario'),
                    'pre_score': record.get('pre_score'),
                    'post_score': record.get('post_score'),
                    'energy_initial': record.get('energy_initial'),
                    'energy_final': record.get('energy_final'),
                    'rsi_accepted': record.get('rsi_accepted'),
                    'plan': record.get('plan')  # stored plan for reuse
                })
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    # Sort by smallest distance
    candidates.sort(key=lambda x: x['distance'])
    return candidates[:top_k]
