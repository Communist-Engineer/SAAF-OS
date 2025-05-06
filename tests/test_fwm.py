#!/usr/bin/env python
"""
Unit tests for the Forward World Model (modules/world_model/fwm.py)
"""

import os
import sys
import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.world_model.fwm import ForwardWorldModel, State, Action, Trajectory, FWMConfig


class TestState(unittest.TestCase):
    """Test cases for the State class."""
    
    def test_initialization(self):
        """Test State initialization."""
        # Create a state vector
        state_vector = np.array([1.0, 2.0, 3.0])
        
        # Create the state
        state = State(state_vector)
        
        # Check attributes
        np.testing.assert_array_equal(state.vector, state_vector)
    
    def test_distance(self):
        """Test distance calculation between states."""
        # Create two states
        state1 = State(np.array([1.0, 2.0, 3.0]))
        state2 = State(np.array([4.0, 5.0, 6.0]))
        
        # Calculate distance (Euclidean)
        distance = state1.distance(state2)
        expected_distance = np.sqrt(np.sum(np.square(np.array([3.0, 3.0, 3.0]))))
        
        # Check result
        self.assertAlmostEqual(distance, expected_distance)
    
    def test_equality(self):
        """Test state equality comparison."""
        # Create states
        state1 = State(np.array([1.0, 2.0, 3.0]))
        state2 = State(np.array([1.0, 2.0, 3.0]))
        state3 = State(np.array([4.0, 5.0, 6.0]))
        
        # Check equality
        self.assertEqual(state1, state2)
        self.assertNotEqual(state1, state3)
        
    def test_to_tensor(self):
        """Test converting state to PyTorch tensor."""
        # Create a state
        state = State(np.array([1.0, 2.0, 3.0]))
        
        # Convert to tensor
        tensor = state.to_tensor()
        
        # Check result
        self.assertIsInstance(tensor, torch.Tensor)
        np.testing.assert_array_equal(tensor.numpy(), state.vector)
    
    def test_from_tensor(self):
        """Test creating state from a PyTorch tensor."""
        # Create a tensor
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        # Create state from tensor
        state = State.from_tensor(tensor)
        
        # Check result
        self.assertIsInstance(state, State)
        np.testing.assert_array_equal(state.vector, tensor.numpy())


class TestAction(unittest.TestCase):
    """Test cases for the Action class."""
    
    def test_initialization(self):
        """Test Action initialization."""
        # Create an action vector
        action_vector = np.array([0.5, -0.3])
        
        # Create the action
        action = Action(action_vector)
        
        # Check attributes
        np.testing.assert_array_equal(action.vector, action_vector)
    
    def test_continuous_action(self):
        """Test continuous action creation."""
        # Create a continuous action
        action = Action.continuous([0.5, -0.3])
        
        # Check attributes
        self.assertIsInstance(action.vector, np.ndarray)
        np.testing.assert_array_equal(action.vector, np.array([0.5, -0.3]))
    
    def test_discrete_action(self):
        """Test discrete action creation."""
        # Create a discrete action
        action = Action.discrete(2, action_space_size=4)
        
        # Check attributes
        self.assertIsInstance(action.vector, np.ndarray)
        np.testing.assert_array_equal(action.vector, np.array([0, 0, 1, 0]))
    
    def test_to_tensor(self):
        """Test converting action to PyTorch tensor."""
        # Create an action
        action = Action(np.array([0.5, -0.3]))
        
        # Convert to tensor
        tensor = action.to_tensor()
        
        # Check result
        self.assertIsInstance(tensor, torch.Tensor)
        np.testing.assert_array_almost_equal(tensor.numpy(), action.vector)
    
    def test_from_tensor(self):
        """Test creating action from a PyTorch tensor."""
        # Create a tensor
        tensor = torch.tensor([0.5, -0.3])
        
        # Create action from tensor
        action = Action.from_tensor(tensor)
        
        # Check result
        self.assertIsInstance(action, Action)
        np.testing.assert_array_almost_equal(action.vector, tensor.numpy())


class TestTrajectory(unittest.TestCase):
    """Test cases for the Trajectory class."""
    
    def test_initialization(self):
        """Test Trajectory initialization."""
        # Create states and actions
        states = [
            State(np.array([1.0, 2.0, 3.0])),
            State(np.array([2.0, 3.0, 4.0])),
            State(np.array([3.0, 4.0, 5.0]))
        ]
        actions = [
            Action(np.array([0.5, -0.3])),
            Action(np.array([0.1, 0.2]))
        ]
        rewards = [1.0, 2.0]
        
        # Create trajectory
        trajectory = Trajectory(states, actions, rewards)
        
        # Check attributes
        self.assertEqual(len(trajectory.states), 3)
        self.assertEqual(len(trajectory.actions), 2)
        self.assertEqual(len(trajectory.rewards), 2)
    
    def test_add_transition(self):
        """Test adding a transition to a trajectory."""
        # Create a trajectory
        trajectory = Trajectory([], [], [])
        
        # Add transitions
        trajectory.add_transition(
            State(np.array([1.0, 2.0, 3.0])),
            Action(np.array([0.5, -0.3])),
            1.0,
            State(np.array([2.0, 3.0, 4.0]))
        )
        
        # Check trajectory was updated
        self.assertEqual(len(trajectory.states), 2)
        self.assertEqual(len(trajectory.actions), 1)
        self.assertEqual(len(trajectory.rewards), 1)
        
        # Add another transition
        trajectory.add_transition(
            State(np.array([2.0, 3.0, 4.0])),
            Action(np.array([0.1, 0.2])),
            2.0,
            State(np.array([3.0, 4.0, 5.0]))
        )
        
        # Check trajectory was updated again
        self.assertEqual(len(trajectory.states), 3)
        self.assertEqual(len(trajectory.actions), 2)
        self.assertEqual(len(trajectory.rewards), 2)
    
    def test_cumulative_reward(self):
        """Test calculating cumulative reward."""
        # Create a trajectory
        trajectory = Trajectory([], [], [])
        
        # Add transitions
        trajectory.add_transition(
            State(np.array([1.0, 2.0, 3.0])),
            Action(np.array([0.5, -0.3])),
            1.0,
            State(np.array([2.0, 3.0, 4.0]))
        )
        trajectory.add_transition(
            State(np.array([2.0, 3.0, 4.0])),
            Action(np.array([0.1, 0.2])),
            2.0,
            State(np.array([3.0, 4.0, 5.0]))
        )
        
        # Calculate cumulative reward
        cumulative = trajectory.cumulative_reward()
        
        # Check result
        self.assertEqual(cumulative, 3.0)
    
    def test_discounted_reward(self):
        """Test calculating discounted reward."""
        # Create a trajectory
        trajectory = Trajectory([], [], [])
        
        # Add transitions
        trajectory.add_transition(
            State(np.array([1.0, 2.0, 3.0])),
            Action(np.array([0.5, -0.3])),
            1.0,
            State(np.array([2.0, 3.0, 4.0]))
        )
        trajectory.add_transition(
            State(np.array([2.0, 3.0, 4.0])),
            Action(np.array([0.1, 0.2])),
            2.0,
            State(np.array([3.0, 4.0, 5.0]))
        )
        
        # Calculate discounted reward (gamma = 0.9)
        discounted = trajectory.discounted_reward(gamma=0.9)
        expected = 1.0 + 0.9 * 2.0
        
        # Check result
        self.assertAlmostEqual(discounted, expected)
    
    def test_to_dataset(self):
        """Test converting trajectory to a training dataset."""
        # Create states and actions
        states = [
            State(np.array([1.0, 2.0, 3.0])),
            State(np.array([2.0, 3.0, 4.0])),
            State(np.array([3.0, 4.0, 5.0]))
        ]
        actions = [
            Action(np.array([0.5, -0.3])),
            Action(np.array([0.1, 0.2]))
        ]
        rewards = [1.0, 2.0]
        
        # Create trajectory
        trajectory = Trajectory(states, actions, rewards)
        
        # Convert to dataset
        X, y = trajectory.to_dataset()
        
        # Check dataset
        self.assertEqual(X.shape[0], 2)  # 2 samples (state-action pairs)
        self.assertEqual(y.shape[0], 2)  # 2 targets (next states)
        
        # Check dataset content
        np.testing.assert_array_equal(X[0], np.concatenate([states[0].vector, actions[0].vector]))
        np.testing.assert_array_equal(y[0], states[1].vector)
        np.testing.assert_array_equal(X[1], np.concatenate([states[1].vector, actions[1].vector]))
        np.testing.assert_array_equal(y[1], states[2].vector)


class TestFWMConfig(unittest.TestCase):
    """Test cases for the FWMConfig class."""
    
    def test_initialization(self):
        """Test FWMConfig initialization with default values."""
        # Create config
        config = FWMConfig()
        
        # Check default values
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.hidden_dim, 128)
        self.assertEqual(config.num_training_iterations, 1000)
        self.assertEqual(config.batch_size, 32)
    
    def test_custom_initialization(self):
        """Test FWMConfig initialization with custom values."""
        # Create config with custom values
        config = FWMConfig(
            learning_rate=0.01,
            hidden_dim=256,
            num_training_iterations=500,
            batch_size=64
        )
        
        # Check custom values
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.hidden_dim, 256)
        self.assertEqual(config.num_training_iterations, 500)
        self.assertEqual(config.batch_size, 64)


class TestForwardWorldModel(unittest.TestCase):
    """Test cases for the ForwardWorldModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Define dimensions for testing
        self.state_dim = 3
        self.action_dim = 2
        
        # Create a model
        self.model = ForwardWorldModel(self.state_dim, self.action_dim)
    
    def test_initialization(self):
        """Test ForwardWorldModel initialization."""
        # Check instance
        self.assertIsInstance(self.model, ForwardWorldModel)
        
        # Check attributes
        self.assertEqual(self.model.state_dim, self.state_dim)
        self.assertEqual(self.model.action_dim, self.action_dim)
        
        # Check if model has neural network components
        self.assertTrue(hasattr(self.model, 'network'))
    
    @patch('torch.save')
    def test_save_model(self, mock_save):
        """Test saving the model."""
        # Test saving
        filename = "test_model.pt"
        self.model.save(filename)
        
        # Check if torch.save was called
        mock_save.assert_called_once()
    
    @patch('torch.load')
    def test_load_model(self, mock_load):
        """Test loading the model."""
        # Mock the load function to return a state dict
        mock_load.return_value = {'state_dict': {}}
        
        # Test loading
        filename = "test_model.pt"
        self.model.load(filename)
        
        # Check if torch.load was called
        mock_load.assert_called_once_with(filename)
    
    def test_predict_next_state(self):
        """Test predicting the next state."""
        # Create a state and action
        state = State(np.zeros(self.state_dim))
        action = Action(np.zeros(self.action_dim))
        
        # Predict next state
        next_state = self.model.predict_next_state(state, action)
        
        # Check result
        self.assertIsInstance(next_state, State)
        self.assertEqual(next_state.vector.shape, (self.state_dim,))
    
    def test_simulate_trajectory(self):
        """Test simulating a trajectory."""
        # Create initial state and a list of actions
        initial_state = State(np.zeros(self.state_dim))
        actions = [Action(np.zeros(self.action_dim)) for _ in range(5)]
        
        # Define a reward function
        def reward_function(state, action, next_state):
            return 1.0
        
        # Simulate trajectory
        trajectory = self.model.simulate_trajectory(initial_state, actions, reward_function)
        
        # Check result
        self.assertIsInstance(trajectory, Trajectory)
        self.assertEqual(len(trajectory.states), len(actions) + 1)
        self.assertEqual(len(trajectory.actions), len(actions))
        self.assertEqual(len(trajectory.rewards), len(actions))
    
    @patch.object(ForwardWorldModel, '_train_step')
    def test_train(self, mock_train_step):
        """Test training the model."""
        # Create some training data
        X = np.random.rand(10, self.state_dim + self.action_dim)
        y = np.random.rand(10, self.state_dim)
        
        # Mock the training step
        mock_train_step.return_value = 0.1
        
        # Train the model
        self.model.train(X, y, num_iterations=5)
        
        # Check if _train_step was called the right number of times
        self.assertEqual(mock_train_step.call_count, 5)
    
    def test_evaluate_mse(self):
        """Test evaluating MSE on test data."""
        # Create some test data
        test_trajectories = [
            Trajectory(
                [State(np.zeros(self.state_dim)) for _ in range(3)],
                [Action(np.zeros(self.action_dim)) for _ in range(2)],
                [1.0 for _ in range(2)]
            )
        ]
        
        # Evaluate
        mse = self.model.evaluate(test_trajectories)
        
        # Check result
        self.assertIsInstance(mse, float)
    
    def test_from_config(self):
        """Test creating a model from config."""
        # Create a config
        config = FWMConfig(
            learning_rate=0.01,
            hidden_dim=256,
            num_training_iterations=500,
            batch_size=64
        )
        
        # Create model from config
        model = ForwardWorldModel.from_config(self.state_dim, self.action_dim, config)
        
        # Check the model
        self.assertIsInstance(model, ForwardWorldModel)
        self.assertEqual(model.config.learning_rate, 0.01)
        self.assertEqual(model.config.hidden_dim, 256)
    
    def test_prediction_error_metrics(self):
        """Test calculating prediction error metrics."""
        # Create actual and predicted states
        actual = State(np.array([1.0, 2.0, 3.0]))
        predicted = State(np.array([1.1, 2.2, 2.8]))
        
        # Calculate metrics
        mae, mse, rmse = self.model.prediction_error_metrics(actual, predicted)
        
        # Expected values
        expected_mae = np.mean(np.abs(actual.vector - predicted.vector))
        expected_mse = np.mean(np.square(actual.vector - predicted.vector))
        expected_rmse = np.sqrt(expected_mse)
        
        # Check results
        self.assertAlmostEqual(mae, expected_mae)
        self.assertAlmostEqual(mse, expected_mse)
        self.assertAlmostEqual(rmse, expected_rmse)


if __name__ == "__main__":
    unittest.main()