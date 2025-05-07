"""
Plan Arbitrator Module for SAAF-OS

This module implements a Plan Arbitration Layer that selects the most promising
plan from multiple candidate strategies by simulating each plan using the Forward
World Model (FWM) and comparing their predicted contradiction scores.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logger = logging.getLogger("SAAF-OS.plan_arbitrator")

def select_best_plan(z_t: np.ndarray, plans: List[Tuple[Dict, str]], fwm) -> Dict:
    """
    Simulates each plan using FWM, returns the one with the lowest predicted contradiction score.
    Logs each plan's origin, energy usage, and score.
    
    Args:
        z_t: Current latent state vector
        plans: List of tuples containing (plan_dict, plan_source) where plan_source is a string
               like "retrieved", "distilled", or "rl"
        fwm: Forward World Model module with simulate_plan function
        
    Returns:
        The plan with the lowest predicted contradiction score
    """
    if not plans:
        raise ValueError("No plans provided to select from")
    
    results = []
    
    # Print header for plan comparison
    print("\nPlan comparison:")
    
    # Evaluate each plan
    for plan, source in plans:
        # Simulate the plan using FWM
        _, contradiction_score = fwm.simulate_plan(z_t, plan)
        
        # Get the plan's energy usage
        energy = plan.get('total_energy', 0.0)
        
        # Store the results for later comparison
        results.append((plan, source, contradiction_score, energy))
        
        # Log the plan's score
        print(f"[{source:<9}] contradiction = {contradiction_score:.3f}, energy = {energy:.2f}")
    
    # Sort plans by contradiction score (ascending)
    sorted_results = sorted(results, key=lambda x: x[2])
    
    # Select the best plan (lowest contradiction score)
    best_plan, best_source, best_score, best_energy = sorted_results[0]
    
    print(f"\n✅ Selected: {best_source}_plan (lowest contradiction)")
    
    return best_plan

def select_best_plan_multimetric(z_t: np.ndarray, plans: List[Tuple[Dict, str]], fwm, 
                               weights: Dict[str, float] = None) -> Dict:
    """
    Extended version of select_best_plan that uses multiple metrics for plan selection.
    
    Args:
        z_t: Current latent state vector
        plans: List of tuples containing (plan_dict, plan_source)
        fwm: Forward World Model module with simulate_plan function
        weights: Dictionary of weights for different metrics:
                - 'contradiction': Weight for contradiction score (default: 0.6)
                - 'energy': Weight for energy efficiency (default: 0.3)
                - 'steps': Weight for plan simplicity (fewer steps is better) (default: 0.1)
        
    Returns:
        The plan with the best combined score
    """
    if not plans:
        raise ValueError("No plans provided to select from")
    
    # Default weights if not provided
    if weights is None:
        weights = {
            'contradiction': 0.6,  # Lower is better
            'energy': 0.3,         # Lower is better
            'steps': 0.1           # Lower is better
        }
    
    results = []
    
    # Print header for plan comparison
    print("\nPlan comparison (multi-metric):")
    
    # Evaluate each plan
    for plan, source in plans:
        # Simulate the plan using FWM
        _, contradiction_score = fwm.simulate_plan(z_t, plan)
        
        # Get the plan metrics
        energy = plan.get('total_energy', 0.0)
        num_steps = len(plan.get('steps', []))
        
        # Normalize steps (assuming max 20 steps for normalization)
        normalized_steps = num_steps / 20.0
        
        # Calculate combined score (lower is better)
        combined_score = (
            contradiction_score * weights['contradiction'] +
            energy * weights['energy'] +
            normalized_steps * weights['steps']
        )
        
        # Store the results
        results.append((plan, source, combined_score, contradiction_score, energy, num_steps))
        
        # Log the plan's details
        print(f"[{source:<9}] combined = {combined_score:.3f}, contradiction = {contradiction_score:.3f}, " +
              f"energy = {energy:.2f}, steps = {num_steps}")
    
    # Sort plans by combined score (ascending)
    sorted_results = sorted(results, key=lambda x: x[2])
    
    # Select the best plan (lowest combined score)
    best_plan, best_source, best_score, best_contra, best_energy, best_steps = sorted_results[0]
    
    print(f"\n✅ Selected: {best_source}_plan (best combined score)")
    
    return best_plan