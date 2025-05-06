#!/usr/bin/env python
"""
Unit tests for the RL Planner (modules/planning/rl_planner.py)
"""

import os
import sys
import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
import pytest

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.planning.rl_planner import RLPlanner, RLConfig, MCTSNode, MCTS


class TestRLConfig(unittest.TestCase):
    """Test cases for the RLConfig class."""
    
    def test_initialization(self):
        """Test RLConfig initialization with default values."""
        config = RLConfig()
        
        # Check default values
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.discount_factor, 0.99)
        self.assertEqual(config.latent_dim, 256)
        self.assertEqual(config.model_hidden_dim, 128)
        self.assertEqual(config.exploration_constant, 1.0)
        self.assertEqual(config.num_simulations, 100)
        self.assertEqual(config.action_space_size, 4)
        self.assertEqual(config.batch_size, 32)
    
    def test_custom_initialization(self):
        """Test RLConfig initialization with custom values."""
        config = RLConfig(
            learning_rate=0.01, 
            discount_factor=0.9, 
            latent_dim=512, 
            model_hidden_dim=256,
            exploration_constant=2.0,
            num_simulations=200,
            action_space_size=6,
            batch_size=64
        )
        
        # Check custom values
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.discount_factor, 0.9)
        self.assertEqual(config.latent_dim, 512)
        self.assertEqual(config.model_hidden_dim, 256)
        self.assertEqual(config.exploration_constant, 2.0)
        self.assertEqual(config.num_simulations, 200)
        self.assertEqual(config.action_space_size, 6)
        self.assertEqual(config.batch_size, 64)


class TestMCTSNode(unittest.TestCase):
    """Test cases for the MCTSNode class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a node for testing
        self.state = np.array([0.5, 0.3, 0.2, 0.1, 0.0])
        self.node = MCTSNode(self.state)
    
    def test_initialization(self):
        """Test node initialization."""
        # Check that the node is properly initialized
        np.testing.assert_array_equal(self.node.state, self.state)
        self.assertEqual(self.node.visit_count, 0)
        self.assertEqual(self.node.value_sum, 0.0)
        self.assertEqual(len(self.node.children), 0)
        self.assertIsNone(self.node.parent)
    
    def test_expand(self):
        """Test node expansion."""
        # Define action priors
        action_priors = [(0, 0.4), (1, 0.3), (2, 0.2), (3, 0.1)]
        
        # Mock the next states
        next_states = [
            np.array([0.52, 0.29, 0.21, 0.11, 0.01]),
            np.array([0.48, 0.32, 0.19, 0.09, 0.02]),
            np.array([0.51, 0.31, 0.18, 0.12, 0.02]),
            np.array([0.49, 0.28, 0.22, 0.08, 0.01])
        ]
        
        # Expand the node
        self.node.expand(action_priors, next_states)
        
        # Check that children were created
        self.assertEqual(len(self.node.children), 4)
        
        # Check that children have correct values
        for action, prior in action_priors:
            child = self.node.children[action]
            self.assertEqual(child.prior, prior)
            self.assertEqual(child.parent, self.node)
            self.assertEqual(child.action, action)
    
    def test_update(self):
        """Test node update."""
        # Initial values
        self.assertEqual(self.node.visit_count, 0)
        self.assertEqual(self.node.value_sum, 0.0)
        
        # Update with a value
        self.node.update(1.5)
        
        # Check updated values
        self.assertEqual(self.node.visit_count, 1)
        self.assertEqual(self.node.value_sum, 1.5)
        
        # Update again
        self.node.update(0.7)
        
        # Check updated values
        self.assertEqual(self.node.visit_count, 2)
        self.assertEqual(self.node.value_sum, 2.2)
        self.assertAlmostEqual(self.node.value(), 1.1)
    
    def test_value(self):
        """Test value calculation."""
        # Empty node should have zero value
        self.assertEqual(self.node.value(), 0.0)
        
        # Update the node
        self.node.visit_count = 3
        self.node.value_sum = 3.6
        
        # Check value calculation
        self.assertAlmostEqual(self.node.value(), 1.2)
    
    def test_ucb_score(self):
        """Test UCB score calculation."""
        # Create a parent node
        parent = MCTSNode(np.zeros(5))
        parent.visit_count = 10
        
        # Create a child node
        child = MCTSNode(np.zeros(5))
        child.parent = parent
        child.prior = 0.3
        child.action = 1
        
        # Empty node should use prior
        score = child.ucb_score(exploration_constant=1.0)
        self.assertGreater(score, 0)
        
        # Update the child
        child.visit_count = 3
        child.value_sum = 2.4
        
        # Calculate UCB score
        score = child.ucb_score(exploration_constant=1.0)
        
        # Expected UCB formula: value + C * prior * sqrt(parent_visits) / (1 + visits)
        expected_score = (2.4 / 3) + 1.0 * 0.3 * (np.sqrt(10) / (1 + 3))
        self.assertAlmostEqual(score, expected_score)
    
    def test_select_child(self):
        """Test child selection based on UCB score."""
        # Create action priors
        action_priors = [(0, 0.4), (1, 0.3), (2, 0.2), (3, 0.1)]
        
        # Mock next states
        next_states = [np.zeros(5) for _ in range(4)]
        
        # Expand the node
        self.node.expand(action_priors, next_states)
        
        # Set visit counts and values for children
        self.node.visit_count = 10
        self.node.children[0].visit_count = 5
        self.node.children[0].value_sum = 3.0
        self.node.children[1].visit_count = 2
        self.node.children[1].value_sum = 2.0
        self.node.children[2].visit_count = 1
        self.node.children[2].value_sum = 1.5
        
        # Select child with exploration constant 1.0
        selected = self.node.select_child(exploration_constant=1.0)
        
        # Due to the UCB formula, child 1 or 2 should be selected (depending on exploration vs exploitation)
        self.assertIn(selected.action, [1, 2, 3])


class TestMCTS(unittest.TestCase):
    """Test cases for the MCTS class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a config for testing
        self.config = RLConfig(
            num_simulations=10,
            exploration_constant=1.0,
            action_space_size=4,
            latent_dim=5
        )
        
        # Mock the forward world model
        self.fwm = MagicMock()
        
        # Create MCTS instance
        self.mcts = MCTS(self.config, self.fwm)
        
        # Mock the contradiction level
        self.contradiction_level = 0.5
        
        # Create an initial state
        self.initial_state = np.array([0.5, 0.3, 0.2, 0.1, 0.0])
    
    def test_initialization(self):
        """Test MCTS initialization."""
        self.assertEqual(self.mcts.config, self.config)
        self.assertEqual(self.mcts.fwm, self.fwm)
    
    def test_search_with_contradictions(self):
        """Test MCTS search with contradictions."""
        # Mock the necessary methods
        self.mcts._simulate = MagicMock(return_value=0.7)
        self.mcts._get_policy_and_values = MagicMock(return_value=(
            np.array([0.25, 0.25, 0.25, 0.25]),  # Prior probabilities
            0.5  # Value
        ))
        self.mcts._get_next_states = MagicMock(return_value=[
            np.array([0.52, 0.29, 0.21, 0.11, 0.01]),
            np.array([0.48, 0.32, 0.19, 0.09, 0.02]),
            np.array([0.51, 0.31, 0.18, 0.12, 0.02]),
            np.array([0.49, 0.28, 0.22, 0.08, 0.01])
        ])
        
        # Run MCTS search
        root = self.mcts.search(self.initial_state, contradiction_level=self.contradiction_level)
        
        # Check that the root node was expanded
        self.assertGreater(root.visit_count, 0)
        self.assertGreaterEqual(len(root.children), 1)
        
        # Check that simulation was called the expected number of times
        self.assertEqual(self.mcts._simulate.call_count, self.config.num_simulations)
    
    def test_contradiction_guided_search(self):
        """Test contradiction-guided search behavior."""
        # Mock the necessary methods
        self.mcts._simulate = MagicMock(return_value=0.7)
        
        # Mock priors for higher contradiction level - should bias toward actions that reduce contradiction
        self.mcts._get_policy_and_values = MagicMock(return_value=(
            np.array([0.1, 0.2, 0.3, 0.4]),  # Prior probabilities - higher for actions that reduce contradiction
            0.5  # Value
        ))
        self.mcts._get_next_states = MagicMock(return_value=[
            np.array([0.52, 0.29, 0.21, 0.11, 0.01]),
            np.array([0.48, 0.32, 0.19, 0.09, 0.02]),
            np.array([0.51, 0.31, 0.18, 0.12, 0.02]),
            np.array([0.49, 0.28, 0.22, 0.08, 0.01])
        ])
        
        # Run MCTS search with high contradiction
        high_root = self.mcts.search(self.initial_state, contradiction_level=0.8)
        
        # Store visit counts for high contradiction
        high_visits = [child.visit_count for child in high_root.children.values()]
        
        # Reset mocks
        self.mcts._simulate.reset_mock()
        
        # Run MCTS search with low contradiction
        low_root = self.mcts.search(self.initial_state, contradiction_level=0.2)
        
        # Store visit counts for low contradiction
        low_visits = [child.visit_count for child in low_root.children.values()]
        
        # With high contradiction, the search should concentrate more on high-prior actions
        # This is a statistical test, so it might occasionally fail, but should generally hold
        high_concentration = sum(v**2 for v in high_visits)
        low_concentration = sum(v**2 for v in low_visits)
        
        # Higher concentration means more visits to fewer nodes
        self.assertGreaterEqual(high_concentration, low_concentration)


class TestRLPlanner(unittest.TestCase):
    """Test cases for the RLPlanner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a config for testing
        self.config = RLConfig(
            latent_dim=5,
            action_space_size=4,
            num_simulations=10
        )
        
        # Create RLPlanner instance - use dummy models for testing
        self.planner = RLPlanner(self.config, use_dummy_models=True)
    
    def test_initialization(self):
        """Test RLPlanner initialization."""
        self.assertEqual(self.planner.config, self.config)
        self.assertTrue(self.planner.use_dummy_models)
        self.assertIsNotNone(self.planner.policy_net)
        self.assertIsNotNone(self.planner.value_net)
        self.assertIsNotNone(self.planner.fwm)
        self.assertIsNotNone(self.planner.mcts)
    
    def test_plan_with_low_contradiction(self):
        """Test planning with low contradiction level."""
        # Prepare inputs
        z_t = np.random.rand(5)
        goal = {"reduce_contradiction": True, "reduce_energy": True}
        
        # Force low contradiction level to test policy network fallback
        z_t = np.array([0.9, 0.1, 0.2, 0.3, 0.4])  # First dimension represents negative contradiction
        
        # Mock policy network to return a specific distribution
        expected_action = 2
        mock_policy = np.zeros(4)
        mock_policy[expected_action] = 1.0
        
        original_get_policy = self.planner._get_policy
        self.planner._get_policy = MagicMock(return_value=mock_policy)
        
        # Mock running MCTS
        original_run_mcts = self.planner._run_mcts
        self.planner._run_mcts = MagicMock()
        
        try:
            # Generate plan
            plan = self.planner.plan(z_t, goal)
            
            # Check that MCTS was not called due to low contradiction
            self.planner._run_mcts.assert_not_called()
            
            # Check that the chosen action matches the expected action
            self.assertEqual(plan["action_type"], expected_action)
            
        finally:
            # Restore original methods
            self.planner._get_policy = original_get_policy
            self.planner._run_mcts = original_run_mcts
    
    def test_plan_with_high_contradiction(self):
        """Test planning with high contradiction level."""
        # Prepare inputs
        z_t = np.array([-0.9, 0.1, 0.2, 0.3, 0.4])  # First dimension represents negative contradiction
        goal = {"reduce_contradiction": True, "reduce_energy": True}
        
        # Mock MCTS to return a specific result
        expected_action = 3
        
        # Create a mock MCTS root node with children that have set visit counts
        mock_root = MCTSNode(z_t)
        mock_root.visit_count = 10
        
        for i in range(4):
            child = MCTSNode(z_t)
            child.parent = mock_root
            child.action = i
            child.visit_count = 2 if i != expected_action else 4
            mock_root.children[i] = child
        
        original_run_mcts = self.planner._run_mcts
        self.planner._run_mcts = MagicMock(return_value=mock_root)
        
        # Mock policy network - should not be used
        original_get_policy = self.planner._get_policy
        self.planner._get_policy = MagicMock()
        
        try:
            # Generate plan
            plan = self.planner.plan(z_t, goal)
            
            # Check that MCTS was called due to high contradiction
            self.planner._run_mcts.assert_called_once()
            
            # Policy should not be called
            self.planner._get_policy.assert_not_called()
            
            # Check that the chosen action matches the expected action
            self.assertEqual(plan["action_type"], expected_action)
            
        finally:
            # Restore original methods
            self.planner._get_policy = original_get_policy
            self.planner._run_mcts = original_run_mcts
    
    def test_hybrid_decision_making(self):
        """Test hybrid decision making with varying contradiction levels."""
        # Prepare a batch of states with different contradiction levels
        states = [
            np.array([0.9, 0.1, 0.2, 0.3, 0.4]),  # Low contradiction (-0.9)
            np.array([0.7, 0.2, 0.3, 0.4, 0.5]),  # Low contradiction (-0.7)
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # High contradiction (-0.1)
            np.array([-0.2, 0.3, 0.4, 0.5, 0.6])   # High contradiction (0.2)
        ]
        
        # Original methods
        original_run_mcts = self.planner._run_mcts
        original_get_policy = self.planner._get_policy
        
        # Count method calls
        mcts_calls = 0
        policy_calls = 0
        
        # Mock methods to track calls
        def mock_run_mcts(state, goal_vector):
            nonlocal mcts_calls
            mcts_calls += 1
            root = MCTSNode(state)
            for i in range(4):
                child = MCTSNode(state)
                child.parent = root
                child.action = i
                child.visit_count = i + 1
                root.children[i] = child
            return root
        
        def mock_get_policy(state):
            nonlocal policy_calls
            policy_calls += 1
            return np.array([0.1, 0.2, 0.3, 0.4])
        
        self.planner._run_mcts = mock_run_mcts
        self.planner._get_policy = mock_get_policy
        
        goal = {"reduce_contradiction": True, "reduce_energy": True}
        
        try:
            # Process each state
            for state in states:
                self.planner.plan(state, goal)
            
            # Check call counts
            # First two states have contradiction_level < 0.1, should use policy
            # Last two states have contradiction_level > 0.1, should use MCTS
            self.assertEqual(mcts_calls, 2)  # Called for high contradiction
            self.assertEqual(policy_calls, 2)  # Called for low contradiction
            
        finally:
            # Restore original methods
            self.planner._run_mcts = original_run_mcts
            self.planner._get_policy = original_get_policy


if __name__ == '__main__':
    unittest.main()