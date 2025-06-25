
# tests/contradiction/test_engine.py

import pytest
from modules.contradiction.engine import ContradictionEngine

def test_evaluate_praxis_outcome():
    """
    Tests that the engine evaluates a successful outcome with a high score.
    """
    engine = ContradictionEngine()
    plan = {"actions": ["action1"]}
    feedback = {"success": True}
    score = engine.evaluate_praxis_outcome(plan, feedback)
    assert score > 0.5

