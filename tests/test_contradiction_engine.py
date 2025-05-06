#!/usr/bin/env python
"""
Unit tests for the Contradiction Engine (modules/contradiction/engine.py)
"""

import os
import sys
import unittest
import networkx as nx
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.contradiction.engine import ContradictionEngine, Node, Edge, ContradictionGraph, SynthesisPlan


class TestNode(unittest.TestCase):
    """Test cases for the Node class."""
    
    def test_initialization(self):
        """Test node initialization."""
        node = Node("goal1", "goal", {"priority": 0.7})
        self.assertEqual(node.id, "goal1")
        self.assertEqual(node.type, "goal")
        self.assertEqual(node.properties, {"priority": 0.7})


class TestEdge(unittest.TestCase):
    """Test cases for the Edge class."""
    
    def test_initialization(self):
        """Test edge initialization."""
        edge = Edge("edge1", "goal1", "goal2", "antagonistic", 0.8)
        self.assertEqual(edge.id, "edge1")
        self.assertEqual(edge.source_id, "goal1")
        self.assertEqual(edge.target_id, "goal2")
        self.assertEqual(edge.contradiction_type, "antagonistic")
        self.assertEqual(edge.tension, 0.8)
        self.assertEqual(edge.history, [])
    
    def test_update_tension(self):
        """Test updating edge tension."""
        edge = Edge("edge1", "goal1", "goal2", "antagonistic", 0.8)
        
        # First update
        edge.update_tension(0.6)
        self.assertEqual(edge.tension, 0.6)
        self.assertEqual(edge.history, [0.8])
        
        # Second update with EWMA
        edge.update_tension(0.4)
        expected_tension = 0.3 * 0.4 + 0.7 * 0.6
        self.assertAlmostEqual(edge.tension, expected_tension)
        self.assertEqual(edge.history, [0.8, 0.6])


class TestContradictionGraph(unittest.TestCase):
    """Test cases for the ContradictionGraph class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph = ContradictionGraph()
        
        # Create test nodes
        self.node1 = Node("goal1", "goal", {"priority": 0.7})
        self.node2 = Node("goal2", "goal", {"priority": 0.8})
        self.node3 = Node("resource1", "resource", {"scarcity": 0.6})
        
        # Create test edges
        self.edge1 = Edge("edge1", "goal1", "goal2", "antagonistic", 0.8)
        self.edge2 = Edge("edge2", "goal1", "resource1", "non-antagonistic", 0.5)
    
    def test_add_node(self):
        """Test adding nodes to the graph."""
        self.graph.add_node(self.node1)
        self.assertEqual(len(self.graph.nodes), 1)
        self.assertEqual(self.graph.nodes["goal1"], self.node1)
        self.assertTrue("goal1" in self.graph.graph)
        
        # Add another node
        self.graph.add_node(self.node2)
        self.assertEqual(len(self.graph.nodes), 2)
    
    def test_add_edge(self):
        """Test adding edges to the graph."""
        # Add nodes first
        self.graph.add_node(self.node1)
        self.graph.add_node(self.node2)
        
        # Add edge
        self.graph.add_edge(self.edge1)
        self.assertEqual(len(self.graph.edges), 1)
        self.assertEqual(self.graph.edges["edge1"], self.edge1)
        self.assertTrue(self.graph.graph.has_edge("goal1", "goal2"))
        
        # Add another edge
        self.graph.add_node(self.node3)
        self.graph.add_edge(self.edge2)
        self.assertEqual(len(self.graph.edges), 2)
    
    def test_update_edge_tension(self):
        """Test updating edge tension."""
        # Add nodes and edge
        self.graph.add_node(self.node1)
        self.graph.add_node(self.node2)
        self.graph.add_edge(self.edge1)
        
        # Original tension
        self.assertEqual(self.graph.edges["edge1"].tension, 0.8)
        
        # Update tension
        self.graph.update_edge_tension("edge1", 0.6)
        self.assertEqual(self.graph.edges["edge1"].tension, 0.6)
        self.assertEqual(self.graph.edges["edge1"].history, [0.8])
        
        # Check that it's updated in the NetworkX graph as well
        self.assertEqual(self.graph.graph["goal1"]["goal2"]["tension"], 0.6)
    
    def test_get_high_tension_edges(self):
        """Test getting high tension edges."""
        # Add nodes and edges
        self.graph.add_node(self.node1)
        self.graph.add_node(self.node2)
        self.graph.add_node(self.node3)
        self.graph.add_edge(self.edge1)  # tension 0.8
        self.graph.add_edge(self.edge2)  # tension 0.5
        
        # Get high tension edges with threshold 0.7
        high_edges = self.graph.get_high_tension_edges(threshold=0.7)
        self.assertEqual(len(high_edges), 1)
        self.assertEqual(high_edges[0].id, "edge1")
        
        # Get high tension edges with threshold 0.4
        high_edges = self.graph.get_high_tension_edges(threshold=0.4)
        self.assertEqual(len(high_edges), 2)
        
        # Get high tension edges with threshold 0.9
        high_edges = self.graph.get_high_tension_edges(threshold=0.9)
        self.assertEqual(len(high_edges), 0)
    
    def test_get_edge_between(self):
        """Test getting edge between nodes."""
        # Add nodes and edges
        self.graph.add_node(self.node1)
        self.graph.add_node(self.node2)
        self.graph.add_node(self.node3)
        self.graph.add_edge(self.edge1)
        self.graph.add_edge(self.edge2)
        
        # Get existing edge
        edge = self.graph.get_edge_between("goal1", "goal2")
        self.assertEqual(edge.id, "edge1")
        
        # Get non-existent edge
        edge = self.graph.get_edge_between("goal2", "resource1")
        self.assertIsNone(edge)


class TestSynthesisPlan(unittest.TestCase):
    """Test cases for the SynthesisPlan class."""
    
    def test_initialization(self):
        """Test synthesis plan initialization."""
        plan = SynthesisPlan("plan1")
        self.assertEqual(plan.id, "plan1")
        self.assertEqual(plan.actions, [])
        self.assertEqual(plan.tension_diff, 0.0)
        self.assertEqual(plan.synthesis_path, [])
    
    def test_add_action(self):
        """Test adding actions to the plan."""
        plan = SynthesisPlan("plan1")
        
        # Add an action
        action1 = {"type": "redistribute_resources", "source_goal": "goal1", "target_goal": "goal2"}
        plan.add_action(action1)
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0], action1)
        
        # Add another action
        action2 = {"type": "optimize_resource", "resource_id": "resource1"}
        plan.add_action(action2)
        self.assertEqual(len(plan.actions), 2)
        self.assertEqual(plan.actions[1], action2)
    
    def test_set_synthesis_path(self):
        """Test setting synthesis path."""
        plan = SynthesisPlan("plan1")
        path = ["goal1", "resource1", "goal2"]
        plan.set_synthesis_path(path)
        self.assertEqual(plan.synthesis_path, path)
    
    def test_set_tension_diff(self):
        """Test setting tension difference."""
        plan = SynthesisPlan("plan1")
        plan.set_tension_diff(-0.3)
        self.assertEqual(plan.tension_diff, -0.3)
    
    def test_to_dict(self):
        """Test converting plan to dictionary."""
        plan = SynthesisPlan("plan1")
        
        # Add data
        plan.add_action({"type": "redistribute_resources"})
        plan.set_synthesis_path(["goal1", "goal2"])
        plan.set_tension_diff(-0.3)
        
        # Convert to dictionary
        plan_dict = plan.to_dict()
        self.assertEqual(plan_dict["plan_id"], "plan1")
        self.assertEqual(len(plan_dict["actions"]), 1)
        self.assertEqual(plan_dict["actions"][0]["type"], "redistribute_resources")
        self.assertEqual(plan_dict["synthesis_path"], ["goal1", "goal2"])
        self.assertEqual(plan_dict["tension_diff"], -0.3)


class TestContradictionEngine(unittest.TestCase):
    """Test cases for the ContradictionEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = ContradictionEngine()
        
        # Create test goals and world state for detecting contradictions
        self.goals = {
            "goal1": {
                "required_resources": ["resource1", "resource2"],
                "priority": 0.8,
                "values": ["efficiency", "productivity"]
            },
            "goal2": {
                "required_resources": ["resource1", "resource3"],
                "priority": 0.7,
                "values": ["sustainability", "equity"]
            },
            "goal3": {
                "required_resources": ["resource4"],
                "priority": 0.6,
                "values": ["productivity"]
            }
        }
        
        self.world_state = {
            "resources": {
                "resource1": {
                    "constrained": True,
                    "scarcity": 0.7,
                    "current_level": 0.3
                },
                "resource2": {
                    "constrained": False,
                    "scarcity": 0.2,
                    "current_level": 0.8
                },
                "resource3": {
                    "constrained": True,
                    "scarcity": 0.5,
                    "current_level": 0.5
                },
                "resource4": {
                    "constrained": False,
                    "scarcity": 0.1,
                    "current_level": 0.9
                }
            },
            "opposing_values": {
                "efficiency": ["equity"],
                "productivity": ["sustainability"]
            }
        }
    
    def test_detect_contradictions(self):
        """Test detecting contradictions between goals and world state."""
        # Detect contradictions
        graph = self.engine.detect_contradictions(self.goals, self.world_state)
        
        # Check that nodes are created for goals and constrained resources
        self.assertIn("goal1", graph.nodes)
        self.assertIn("goal2", graph.nodes)
        self.assertIn("goal3", graph.nodes)
        self.assertIn("resource1", graph.nodes)  # Constrained
        self.assertIn("resource3", graph.nodes)  # Constrained
        
        # Check that edges are created for resource conflicts
        edge = graph.get_edge_between("goal1", "goal2")
        self.assertIsNotNone(edge)
        self.assertGreater(edge.tension, 0)
        
        # Check that edges are created for value conflicts
        value_edge = graph.get_edge_between("goal1", "goal2")
        self.assertIsNotNone(value_edge)
        self.assertEqual(value_edge.contradiction_type, "antagonistic")
    
    def test_resolve_contradiction(self):
        """Test resolving contradictions."""
        # First, detect contradictions
        graph = self.engine.detect_contradictions(self.goals, self.world_state)
        
        # Then, resolve contradiction between goal1 and goal2
        plan = self.engine.resolve_contradiction("goal1", "goal2")
        
        # Check that a plan is generated
        self.assertIsNotNone(plan)
        self.assertTrue(len(plan.actions) > 0)
        self.assertTrue(plan.tension_diff < 0)  # Should reduce tension
        
        # Check that the plan includes a path
        self.assertTrue(len(plan.synthesis_path) > 0)
        self.assertEqual(plan.synthesis_path[0], "goal1")
        
        # Verify the action type based on node types
        self.assertEqual(plan.actions[0]["type"], "redistribute_resources")
    
    def test_evaluate_praxis_outcome(self):
        """Test evaluating the outcome of a synthesis plan."""
        # Create a synthesis plan
        plan = SynthesisPlan("test_plan")
        plan.set_synthesis_path(["goal1", "resource1", "goal2"])
        plan.set_tension_diff(-0.4)
        
        # Create feedback data
        feedback = {
            "success_rate": 0.8,
            "tension_reduction": 0.6,
            "labor_time": 30,
            "energy_usage": 20,
            "node_updates": {
                "goal1": {"priority": 0.75},
                "goal2": {"priority": 0.65}
            }
        }
        
        # First detect contradictions to populate the graph
        self.engine.detect_contradictions(self.goals, self.world_state)
        
        # Evaluate the outcome
        score = self.engine.evaluate_praxis_outcome(plan, feedback)
        
        # Check that a score is calculated
        self.assertGreater(score, 0)
        self.assertLess(score, 1)
        
        # Check that the node properties were updated
        goal1_node = self.engine.contradiction_graph.get_node("goal1")
        self.assertEqual(goal1_node.properties["priority"], 0.75)


if __name__ == '__main__':
    unittest.main()