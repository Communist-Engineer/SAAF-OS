"""
Scenario definitions for SAAF-OS simulations.
"""
from typing import Dict, Any, List, Tuple

def load_scenario(name: str) -> Dict[str, Any]:
    """
    Load a predefined synthetic scenario by name.

    Returns a dict with keys:
      - u_t: world state dict
      - goal: dict specifying planning goal parameters
      - contradictions: List of tuples (actor1, actor2, tension)
      - description: Optional scenario description
      - priority_level: Optional numeric priority
    """
    scenarios = {
        "solar_conflict": {
            "u_t": {
                "solar": 50,
                "demand": [20, 30, 25],
                "resources": {"water": 60},
                "priority": 0.8
            },
            "goal": {"target_energy": 0.6, "priority": 0.8, "description": "Resolve solar power contention"},
            "contradictions": [("AgriBot1", "AgriBot2", 0.5), ("AgriBot2", "FabBot", 0.4)],
            "description": "Multiple bots draw from limited solar energy at peak demand.",
            "priority_level": 0.9
        },
        "veto_loop": {
            "u_t": {
                "energy_grid": 70,
                "demand": [40, 35],
                "resources": {"battery": 20},
                "priority": 0.5
            },
            "goal": {"target_energy": 0.5, "priority": 0.5, "description": "Test governance veto scenarios"},
            "contradictions": [("RSIEngine", "Governance", 0.7)],
            "description": "RSI proposes risky patch that governance may override.",
            "priority_level": 0.6
        },
        "alienation_drift": {
            "u_t": {
                "labor_capacity": 30,
                "intention_score": 0.3,
                "resources": {"tools": 10},
                "priority": 0.4
            },
            "goal": {"target_energy": 0.4, "priority": 0.4, "description": "Align labor with intention"},
            "contradictions": [("Planner", "Agent", 0.6)],
            "description": "Planner-agent mismatch in task allocation causes drift.",
            "priority_level": 0.7
        }
    }
    return scenarios.get(name)
