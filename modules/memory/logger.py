import json
import os
from typing import Optional, Dict, Any, List
import numpy as np

def log_episode(
    scenario_name: str,
    inputs: Dict[str, Any],
    plan: Dict[str, Any],
    z_t: np.ndarray,
    z_t_prime: np.ndarray,
    pre_score: float,
    post_score: float,
    rsi_patch: Optional[str],
    accepted: bool,
    output_path: str = "memory/episodes.jsonl"
) -> None:
    """
    Log an episode to a JSONL file.

    Each line is a JSON object with episode data.
    """
    # Ensure output directory exists
    dir_path = os.path.dirname(output_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    # Compute latent delta
    delta = z_t_prime - z_t
    # Get top-3 changed dimensions
    indices = list(np.argsort(np.abs(delta))[-3:])
    top_changes: List[Dict[str, Any]] = []
    for idx in indices:
        top_changes.append({
            "dim": int(idx),
            "delta": float(delta[idx])
        })

    record: Dict[str, Any] = {
        "scenario": scenario_name,
        "inputs": inputs,
        "plan": plan,
        "pre_score": pre_score,
        "post_score": post_score,
        "z_t": z_t.tolist(),
        "z_t_prime": z_t_prime.tolist(),
        "rsi_patch": rsi_patch,
        "rsi_accepted": accepted,
        "latent_summary": top_changes,
        "energy_initial": plan.get("total_energy_before", plan.get("total_energy")),
        "energy_final": plan.get("total_energy"),
    }

    # Append as JSON line
    with open(output_path, "a") as f:
        f.write(json.dumps(record))
        f.write("\n")
