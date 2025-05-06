"""
Simple Plan Generator for SAAF-OS

This module implements a simplified plan generator that creates dummy plans
and potential contradictions based on a latent state and goal.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

def generate_plan(z_t: np.ndarray, goal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a simple plan based on the latent state and goal.
    
    Args:
        z_t: Latent state vector
        goal: Dictionary containing goal specifications
        
    Returns:
        Dictionary containing the plan and potential contradictions
    """
    # Extract goal parameters (with defaults if not specified)
    target_energy = goal.get('target_energy', 0.7)
    priority = goal.get('priority', 0.5)
    
    # Create a list of steps (more steps for higher priority goals)
    num_steps = 2 + int(priority * 3)
    steps = []
    
    # Generate plan steps based on latent state and goal
    for i in range(num_steps):
        # Create steps with varying energy requirements
        energy_per_step = target_energy / num_steps
        
        step = {
            'id': f'step_{i+1}',
            'action': _get_action_by_index(i),
            'energy_required': energy_per_step * (0.8 + 0.4 * np.random.random()),
            'duration': int(10 + 20 * np.random.random()),
            'agent_id': _get_agent_by_index(i % 3)
        }
        steps.append(step)
    
    # Calculate total energy usage
    total_energy = sum(step['energy_required'] for step in steps)
    
    # Generate potential contradictions between actors
    contradictions = _generate_contradictions(steps)
    
    # Create and return the plan
    return {
        'plan': {
            'steps': steps,
            'total_energy': total_energy,
            'agent_id': steps[0]['agent_id'] if steps else 'AgriBot1',
            'estimated_completion_time': sum(step['duration'] for step in steps)
        },
        'contradictions': contradictions
    }

def _get_action_by_index(idx: int) -> str:
    """Return an action name based on index."""
    actions = [
        'harvest', 'water_crops', 'fertilize', 
        'analyze_soil', 'prune', 'plant_seeds',
        'monitor_pests', 'distribute_resources', 'recharge'
    ]
    return actions[idx % len(actions)]

def _get_agent_by_index(idx: int) -> str:
    """Return an agent name based on index."""
    agents = ['AgriBot1', 'AgriBot2', 'FabBot']
    return agents[idx % len(agents)]

def _generate_contradictions(steps: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
    """
    Generate potential contradictions between actors in the plan.
    
    Args:
        steps: List of plan steps
        
    Returns:
        List of tuples (actor1, actor2, tension_value) representing contradictions
    """
    contradictions = []
    agents = set(step['agent_id'] for step in steps)
    
    # Calculate energy usage per agent
    agent_energy = {}
    for step in steps:
        agent_id = step['agent_id']
        if agent_id not in agent_energy:
            agent_energy[agent_id] = 0
        agent_energy[agent_id] += step['energy_required']
    
    # Generate contradictions between agents with higher energy usage
    agents_list = list(agents)
    for i in range(len(agents_list)):
        for j in range(i+1, len(agents_list)):
            agent1 = agents_list[i]
            agent2 = agents_list[j]
            
            # Calculate tension based on energy usage difference
            energy1 = agent_energy.get(agent1, 0)
            energy2 = agent_energy.get(agent2, 0)
            
            # Higher combined energy and larger difference creates more tension
            tension = (energy1 + energy2) * 0.1 + abs(energy1 - energy2) * 0.3
            
            # Add some randomness to tension
            tension = min(1.0, max(0.0, tension + (np.random.random() * 0.2 - 0.1)))
            
            # Only add significant contradictions
            if tension > 0.3:
                contradictions.append((agent1, agent2, tension))
    
    return contradictions