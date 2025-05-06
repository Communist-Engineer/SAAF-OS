"""
Test cases for the simplified ULS encoder module.
"""

import sys
import os
import pytest
import numpy as np

# Add the repository root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our ULS encoder
from modules.uls_encoder import encode_state

def test_encode_state():
    """Test that encode_state produces a 16-dimensional vector."""
    # Simple dictionary input
    z = encode_state({"solar": 100, "demand": [30, 50]})
    assert z.shape == (16,)
    
    # Check the vector is normalized
    assert np.isclose(np.linalg.norm(z), 1.0)
    
    # Test with nested dictionary
    z2 = encode_state({
        "resources": {
            "solar": 80, 
            "water": 50
        },
        "agents": [1, 2, 3, 4]
    })
    assert z2.shape == (16,)
    assert np.isclose(np.linalg.norm(z2), 1.0)
    
    # Test with empty input (should still return a vector with random noise)
    z3 = encode_state({})
    assert z3.shape == (16,)
    assert np.isclose(np.linalg.norm(z3), 1.0)

def test_encode_state_deterministic():
    """Test that encode_state is deterministic for the same input."""
    # Same input should produce same output (except for random noise part)
    input_data = {"solar": 100, "demand": [30, 50], "priority": 0.8}
    z1 = encode_state(input_data)
    z2 = encode_state(input_data)
    
    # First few dimensions should be consistent (deterministic part)
    # We only check the first few since the remaining might have random noise
    assert np.allclose(z1[:3], z2[:3])
    
def test_encode_state_with_different_inputs():
    """Test that encode_state produces different outputs for different inputs."""
    z1 = encode_state({"solar": 100, "demand": [30, 50]})
    z2 = encode_state({"solar": 200, "demand": [30, 50]})
    
    # The vectors should be different (due to different solar value)
    assert not np.allclose(z1, z2)
    
if __name__ == "__main__":
    # Run tests manually
    test_encode_state()
    test_encode_state_deterministic()
    test_encode_state_with_different_inputs()
    print("All tests passed!")