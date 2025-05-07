"""
Simplified Unified Latent Space (ULS) Encoder for SAAF-OS

This module implements a simplified ULS Encoder that converts raw system state (u_t) 
into a unified latent representation (z_t) with configurable dimensions.
"""

import numpy as np
from typing import Dict, Any, List, Union

def encode_state(u_t: dict, output_dim: int = 16) -> np.ndarray:
    """
    Encode raw system state (u_t) into a unified latent space (z_t).
    
    Args:
        u_t: Dictionary containing numerical or list inputs (e.g., power levels, goals)
            Expected keys can include 'solar', 'demand', and other system state metrics
        output_dim: Dimension of the output vector (default: 16)
    
    Returns:
        z_t: A NumPy vector representing the state in latent space with the specified dimension
    """
    # Initialize a zero vector with the specified dimension
    z_t = np.zeros(output_dim)
    
    # Fill the vector with values based on u_t keys
    idx = 0
    
    # Process scalar values
    for key, value in u_t.items():
        if isinstance(value, (int, float)):
            if idx < output_dim:
                # Normalize large values
                if value > 100:
                    z_t[idx] = value / 1000
                else:
                    z_t[idx] = value / 10
                idx += 1
        elif isinstance(value, dict):
            # For nested dictionaries, extract values
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (int, float)) and idx < output_dim:
                    z_t[idx] = subvalue / 10
                    idx += 1
    
    # Process lists/arrays separately to ensure they're concatenated properly
    for key, value in u_t.items():
        if isinstance(value, (list, np.ndarray)):
            # Convert to numpy array if it's a list
            arr = np.array(value) if isinstance(value, list) else value
            
            # Flatten if multi-dimensional
            flat_arr = arr.flatten()
            
            # Calculate how many values we can fit
            remaining_space = output_dim - idx
            elements_to_add = min(len(flat_arr), remaining_space)
            
            # Add as many values as possible
            if elements_to_add > 0:
                z_t[idx:idx+elements_to_add] = flat_arr[:elements_to_add] / 100  # Normalize
                idx += elements_to_add
    
    # If we couldn't fill all dimensions, add some noise to remaining dimensions
    if idx < output_dim:
        z_t[idx:] = np.random.randn(output_dim - idx) * 0.01  # Small random noise
    
    # Normalize the vector to unit length for consistent representation
    norm = np.linalg.norm(z_t)
    if norm > 0:
        z_t = z_t / norm
    
    return z_t