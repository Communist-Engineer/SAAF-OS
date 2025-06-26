# tests/contradiction/test_engine.py

import pytest
from modules.contradiction.engine import ContradictionEngine

@pytest.fixture
def engine():
    """Provides a fresh ContradictionEngine for each test."""
    return ContradictionEngine()

def test_initialization(engine):
    """Tests that the engine initializes with an empty graph."""
    assert engine.get_graph().number_of_nodes() == 0
    assert engine.get_graph().number_of_edges() == 0

def test_detect_antagonistic_contradiction(engine):
    """Tests detection of a direct 'increase' vs. 'decrease' contradiction."""
    nodes = {
        "node1": {"entity": "metric_A", "action": "increase"},
        "node2": {"entity": "metric_A", "action": "decrease"}
    }
    graph = engine.detect_contradictions(nodes, {})
    assert graph.has_edge("node1", "node2")
    edge_data = graph.get_edge_data("node1", "node2")
    assert edge_data["type"] == "antagonistic"
    assert edge_data["tension"] > 0.5

def test_detect_synergistic_relation(engine):
    """Tests detection of a synergistic relationship (e.g., two 'increase' nodes)."""
    nodes = {
        "node1": {"entity": "metric_A", "action": "increase"},
        "node2": {"entity": "metric_A", "action": "increase"}
    }
    graph = engine.detect_contradictions(nodes, {})
    assert graph.has_edge("node1", "node2")
    edge_data = graph.get_edge_data("node1", "node2")
    assert edge_data["type"] == "synergistic"
    assert edge_data["tension"] < 0

def test_no_contradiction_for_unrelated_goals(engine):
    """Tests that no edge is created for nodes affecting different entities."""
    nodes = {
        "node1": {"entity": "metric_A", "action": "increase"},
        "node2": {"entity": "metric_B", "action": "decrease"}
    }
    graph = engine.detect_contradictions(nodes, {})
    assert not graph.has_edge("node1", "node2")

def test_resolve_contradiction_with_astar(engine):
    """Tests the A* pathfinding for resolving contradictions."""
    nodes = {
        "n1": {"entity": "A", "action": "increase", "pos": (0, 0)},
        "n2": {"entity": "B", "action": "neutral", "pos": (1, 1)},
        "n3": {"entity": "C", "action": "decrease", "pos": (2, 0)}
    }
    # Manually build a graph for this test
    engine.graph.add_node("n1", pos=(0,0))
    engine.graph.add_node("n2", pos=(1,1))
    engine.graph.add_node("n3", pos=(2,0))
    engine.graph.add_edge("n1", "n2", tension=1.0, feasibility_cost=0.5)
    engine.graph.add_edge("n2", "n3", tension=1.0, feasibility_cost=0.5)

    plan = engine.resolve_contradiction("n1", "n3")
    assert "error" not in plan
    assert len(plan["actions"]) == 2
    assert plan["estimated_tension_reduction"] == 3.0

def test_resolve_contradiction_no_path(engine):
    """Tests the case where no path exists between two nodes."""
    engine.graph.add_node("g1")
    engine.graph.add_node("g2")
    plan = engine.resolve_contradiction("g1", "g2")
    assert "error" in plan
    assert "No synthesis path found" in plan["error"]

def test_evaluate_praxis_outcome_success(engine):
    """Tests that the engine evaluates a successful outcome with a high score based on multi-objective feedback."""
    plan = {"actions": ["action1"]}
    feedback = {
        "labor_time_saved": 0.9,
        "energy_efficiency_gain": 0.8,
        "class_alignment_score": 0.95,
        "success": True
    }
    score = engine.evaluate_praxis_outcome(plan, feedback)
    # Expected score: 0.4 * 0.9 + 0.3 * 0.8 + 0.3 * 0.95 = 0.36 + 0.24 + 0.285 = 0.885
    assert abs(score - 0.885) < 1e-9

def test_evaluate_praxis_outcome_failure(engine):
    """Tests that the engine evaluates a failed outcome with a low score due to overall failure."""
    plan = {"actions": ["action1"]}
    feedback = {
        "labor_time_saved": 0.5,
        "energy_efficiency_gain": 0.5,
        "class_alignment_score": 0.5,
        "success": False
    }
    score = engine.evaluate_praxis_outcome(plan, feedback)
    # Expected score: (0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 0.5) * 0.5 = (0.2 + 0.15 + 0.15) * 0.5 = 0.5 * 0.5 = 0.25
    assert abs(score - 0.25) < 1e-9

def test_update_tension_ewma(engine):
    """Tests that the tension is updated using EWMA."""
    engine.graph.add_edge("g1", "g2", tension=0.5, history=[0.5])
    engine.update_tension("g1", "g2", 1.0)
    new_tension = engine.graph["g1"]["g2"]["tension"]
    # Expected: 0.3 * 1.0 + (1 - 0.3) * 0.5 = 0.3 + 0.35 = 0.65
    assert abs(new_tension - 0.65) < 1e-9
