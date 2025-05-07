"""
Utilities for planner plan representation and vectorization.
"""
import numpy as np
from typing import Dict, Any

def vectorize_plan(plan: Dict[str, Any]) -> np.ndarray:
    """
    Encode a plan dictionary into a fixed-length numeric vector.
    Features: [total_energy, num_steps, avg_step_energy, unique_agents]
    """
    total_energy = plan.get('total_energy', 0.0)
    steps = plan.get('steps', [])
    num_steps = len(steps)
    if num_steps > 0:
        avg_step_energy = total_energy / num_steps
    else:
        avg_step_energy = 0.0
    agents = {step.get('agent_id') for step in steps}
    unique_agents = len(agents)
    # vector of size 4
    return np.array([total_energy, num_steps, avg_step_energy, unique_agents], dtype=float)

def reconstruct_plan_from_vector(vec: np.ndarray) -> Dict[str, Any]:
    """
    Convert a vectorized plan representation back into a dummy plan dictionary.
    """
    total_energy, num_steps, avg_step_energy, unique_agents = vec.tolist()
    num_steps = max(1, int(round(num_steps)))
    agents = ['AgriBot1', 'AgriBot2', 'FabBot']
    # Use as many distinct agents as indicated
    unique_agents = max(1, int(round(unique_agents)))
    steps = []
    for i in range(num_steps):
        steps.append({
            'id': f'pred_step_{i}',
            'action': 'predicted',
            'energy_required': float(avg_step_energy),
            'duration': 1,
            'agent_id': agents[i % unique_agents]
        })
    return {
        'steps': steps,
        'total_energy': float(total_energy),
        'estimated_completion_time': num_steps
    }
