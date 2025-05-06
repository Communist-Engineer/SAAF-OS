"""
Forward World Model (FWM) Module for SAAF-OS

This module implements a simplified Forward World Model that simulates 
the effects of applying a plan to a latent state.
"""

import numpy as np
from typing import Dict, Tuple, List, Any

def simulate_plan(z_t: np.ndarray, plan: Dict) -> Tuple[np.ndarray, float]:
    """
    Simulate applying a plan to a latent state z_t.
    
    Args:
        z_t: Current latent state vector
        plan: Dictionary containing plan details including steps, energy usage, etc.
            Expected keys: 'steps', 'total_energy'
    
    Returns:
        Tuple containing:
        - Predicted latent state z_t+1
        - Simulated contradiction score (float)
    """
    # Create a copy of the input state to avoid modifying the original
    z_next = z_t.copy()
    
    # Extract plan features
    total_energy = plan.get('total_energy', 0.0)
    num_steps = len(plan.get('steps', []))
    
    # Extract unique agents involved in the plan
    agents = set()
    for step in plan.get('steps', []):
        agents.add(step.get('agent_id', 'unknown'))
    num_agents = len(agents)
    
    # Calculate complexity factor based on number of steps and agents
    complexity = num_steps * 0.1 * num_agents * 0.3
    
    # Simulate the effects on the latent vector
    # Higher energy consumption increases the contradiction (dimension 0)
    z_next[0] -= total_energy * 0.2
    
    # More steps make the plan more complex (dimension 1)
    z_next[1] += complexity * 0.15
    
    # Energy efficiency (dimension 2) decreases with higher energy use
    z_next[2] -= total_energy * 0.1
    
    # Resource utilization (dimension 3) increases
    z_next[3] += total_energy * 0.25
    
    # Add some small random perturbations to other dimensions
    # to simulate unpredictable effects (limited to dimensions 4-15)
    if len(z_next) > 4:
        perturbation = np.random.randn(min(len(z_next) - 4, 12)) * 0.05
        z_next[4:4+len(perturbation)] += perturbation
    
    # Ensure the vector remains normalized
    z_next = z_next / np.linalg.norm(z_next)
    
    # Calculate contradiction score based on the new state
    # Lower values in first dimension indicate higher contradiction
    contradiction_score = 0.5 - z_next[0]
    
    # Ensure score is in [0, 1] range
    contradiction_score = max(0.0, min(1.0, contradiction_score))
    
    return z_next, contradiction_score

def batch_simulate_plans(z_t: np.ndarray, plans: List[Dict]) -> List[Tuple[np.ndarray, float]]:
    """
    Simulate applying multiple plans to the same latent state.
    
    Args:
        z_t: Current latent state vector
        plans: List of plan dictionaries
    
    Returns:
        List of (next_state, contradiction_score) tuples
    """
    return [simulate_plan(z_t, plan) for plan in plans]

def compare_plans(z_t: np.ndarray, plans: List[Dict]) -> Dict[str, Any]:
    """
    Compare multiple plans by simulating and ranking them.
    
    Args:
        z_t: Current latent state vector
        plans: List of plan dictionaries, each with an 'id' key
    
    Returns:
        Dictionary with rankings and results
    """
    results = []
    
    for i, plan in enumerate(plans):
        plan_id = plan.get('id', f"plan_{i}")
        z_next, contradiction_score = simulate_plan(z_t, plan)
        
        # Calculate a simple utility score (lower contradiction is better)
        # Also consider energy efficiency (higher is better)
        utility = (1.0 - contradiction_score) * 0.7 + (z_next[2] + 1.0) / 2.0 * 0.3
        
        results.append({
            'plan_id': plan_id,
            'utility': utility,
            'contradiction_score': contradiction_score,
            'z_next': z_next.tolist()
        })
    
    # Sort results by utility (descending)
    sorted_results = sorted(results, key=lambda x: x['utility'], reverse=True)
    
    return {
        'best_plan_id': sorted_results[0]['plan_id'] if sorted_results else None,
        'plans': sorted_results
    }