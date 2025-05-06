"""
Hybrid RL Planner for SAAF-OS

This module implements the hybrid reinforcement learning planner as specified in rl_loop_spec.md.
It combines direct policy optimization with Monte Carlo Tree Search (MCTS) in the Unified Latent Space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import heapq
import logging
import math
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RLPlanner")


class PolicyNetwork(nn.Module):
    """
    Policy network that maps latent states to actions as specified in rl_loop_spec.md.
    """
    
    def __init__(self, latent_dim: int = 256, action_dim: int = 32, hidden_dim: int = 128):
        """
        Initialize the policy network.
        
        Args:
            latent_dim: Dimension of the latent state
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network (for actor-critic methods)
        self.value = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the policy network.
        
        Args:
            z_t: Latent state [batch_size, latent_dim]
            
        Returns:
            Tuple of:
                action_logits: Action logits [batch_size, action_dim]
                value: Value prediction [batch_size, 1]
        """
        action_logits = self.policy(z_t)
        value = self.value(z_t)
        
        return action_logits, value


@dataclass
class MCTSConfig:
    """Configuration for Monte Carlo Tree Search."""
    num_simulations: int = 50
    exploration_weight: float = 1.0
    discount_factor: float = 0.95
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25


@dataclass
class RLConfig:
    """Configuration for reinforcement learning."""
    learning_rate: float = 1e-4
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.01
    clip_grad_norm: float = 1.0
    batch_size: int = 64
    mcts_config: MCTSConfig = MCTSConfig()


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree.
    """
    
    def __init__(self, state: np.ndarray, prior: float = 0.0):
        """
        Initialize an MCTS node.
        
        Args:
            state: The state represented by this node
            prior: Prior probability of selecting this node
        """
        self.state = state  # Latent state z_t
        self.prior = prior  # Prior probability from policy
        
        # Tree search statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # Map from action to child node
        self.reward = 0.0  # Immediate reward for reaching this node
        self.contradiction_score = 0.0  # L_contradiction score
        
    def value(self) -> float:
        """
        Get the average value of this node.
        
        Returns:
            Average value
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def expanded(self) -> bool:
        """
        Check if this node has been expanded.
        
        Returns:
            True if expanded, False otherwise
        """
        return len(self.children) > 0
    
    def add_child(self, action: int, child_node):
        """
        Add a child node.
        
        Args:
            action: Action leading to the child
            child_node: The child node
        """
        self.children[action] = child_node


class MCTS:
    """
    Monte Carlo Tree Search in latent space as specified in rl_loop_spec.md.
    """
    
    def __init__(self, 
                 config: MCTSConfig,
                 forward_model,
                 contradiction_scorer: Callable[[np.ndarray], float],
                 action_space_size: int = 32):
        """
        Initialize the MCTS.
        
        Args:
            config: MCTS configuration
            forward_model: Function to predict next state and reward given state and action
            contradiction_scorer: Function to calculate contradiction loss for a state
            action_space_size: Number of possible actions
        """
        self.config = config
        self.forward_model = forward_model
        self.contradiction_scorer = contradiction_scorer
        self.action_space_size = action_space_size
    
    def run(self, root_state: np.ndarray, model: PolicyNetwork) -> Dict[int, float]:
        """
        Run MCTS from a root state.
        
        Args:
            root_state: The root state to start from
            model: Policy network for action probabilities
            
        Returns:
            Dictionary mapping actions to visit counts
        """
        # Create root node
        root = MCTSNode(root_state)
        
        # Expand root node
        self._expand_node(root, model)
        
        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            
            # Select leaf node
            while node.expanded():
                action, node = self._select_child(node)
                search_path.append(node)
            
            # Expand leaf node and evaluate
            value = 0.0
            if node.visit_count > 0:  # Skip expansion for already visited nodes
                self._expand_node(node, model)
                # Use the contradiction score as a negative reward
                value = -node.contradiction_score
            else:
                # Evaluate leaf node directly
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(node.state).unsqueeze(0)
                    _, value_tensor = model(state_tensor)
                    value = value_tensor.item()
            
            # Backpropagate
            self._backpropagate(search_path, value)
        
        # Return normalized visit counts as action probabilities
        visit_counts = {a: n.visit_count for a, n in root.children.items()}
        total_visits = sum(visit_counts.values())
        return {a: count / total_visits for a, count in visit_counts.items()}
    
    def _expand_node(self, node: MCTSNode, model: PolicyNetwork) -> None:
        """
        Expand a node by adding all possible children.
        
        Args:
            node: The node to expand
            model: Policy network for action probabilities
        """
        # Get action probabilities from policy network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(node.state).unsqueeze(0)
            action_logits, _ = model(state_tensor)
            action_probs = F.softmax(action_logits, dim=-1).squeeze(0).numpy()
        
        # Add Dirichlet noise for exploration
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * self.action_space_size)
        action_probs = (1 - self.config.dirichlet_weight) * action_probs + self.config.dirichlet_weight * noise
        
        # Calculate contradiction score for current state
        node.contradiction_score = self.contradiction_scorer(node.state)
        
        # Create children for all actions
        for action in range(self.action_space_size):
            # Use forward model to predict next state
            next_state, reward = self.forward_model(node.state, action)
            
            # Create child node
            child = MCTSNode(next_state, prior=action_probs[action])
            child.reward = reward
            
            # Add child
            node.add_child(action, child)
    
    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """
        Select a child node according to the PUCT formula.
        
        Args:
            node: The parent node
            
        Returns:
            Tuple of selected action and child node
        """
        # PUCT (Predictor + Upper Confidence bound for Trees)
        exploration_factor = self.config.exploration_weight * math.sqrt(node.visit_count)
        
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            # PUCT formula
            exploit = child.value()
            explore = child.prior * exploration_factor / (1 + child.visit_count)
            
            # Add contradiction gradient term to guide search
            contradiction_term = -0.5 * child.contradiction_score
            
            # Combined score
            score = exploit + explore + contradiction_term
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _backpropagate(self, search_path: List[MCTSNode], value: float) -> None:
        """
        Backpropagate value through the search path.
        
        Args:
            search_path: List of visited nodes
            value: Value to backpropagate
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            
            # Incorporate reward and discount for next iteration
            if node != search_path[-1]:  # Not the leaf node
                value = node.reward + self.config.discount_factor * value


class DummyForwardModel:
    """
    Simple forward model for MCTS that doesn't require a full neural network.
    Used for prototype testing.
    """
    
    def __init__(self, latent_dim: int = 256):
        """
        Initialize the dummy forward model.
        
        Args:
            latent_dim: Dimension of the latent state
        """
        self.latent_dim = latent_dim
    
    def __call__(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """
        Predict next state and reward given state and action.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Tuple of next state and reward
        """
        # Create a deterministic but action-dependent perturbation
        np.random.seed(hash(str(action)) % (2**32))
        
        # Create perturbation
        perturbation = np.random.randn(self.latent_dim) * 0.1
        
        # Make certain actions have specific effects
        if action % 4 == 0:  # Action type 0 - redistribute resources
            perturbation[0] = 0.3  # Reduce contradiction
        elif action % 4 == 1:  # Action type 1 - optimize resources
            perturbation[1] = -0.2  # Reduce energy usage
        elif action % 4 == 2:  # Action type 2 - mediate
            perturbation[0] = 0.1  # Slightly reduce contradiction
        
        # Apply perturbation
        next_state = state + perturbation
        next_state = next_state / np.linalg.norm(next_state)
        
        # Calculate reward based on contradictions and other metrics
        reward = 0.5 * next_state[0]  # Higher is better (less contradiction)
        reward -= 0.3 * next_state[1]  # Lower energy usage is better
        
        return next_state, reward


class DummyPolicyNetwork:
    """
    A simple policy network that doesn't require a full neural network.
    Used for prototype testing.
    """
    
    def __init__(self, action_space_size: int = 32):
        """
        Initialize the dummy policy network.
        
        Args:
            action_space_size: Number of possible actions
        """
        self.action_space_size = action_space_size
    
    def __call__(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict action probabilities and value given state.
        
        Args:
            state: Current state
            
        Returns:
            Tuple of action probabilities and value
        """
        # Create a deterministic but state-dependent output
        np.random.seed(hash(str(state.tolist())) % (2**32))
        
        # Generate action probabilities
        logits = np.random.randn(self.action_space_size)
        
        # Bias towards actions that reduce contradiction if indicated in the state
        if state[0] < 0:  # First dimension tracks contradiction
            logits[0] += 2.0  # Bias towards "redistribute_resources"
        
        # Bias towards energy optimization if energy usage is high
        if state[1] > 0.5:
            logits[1] += 1.5  # Bias towards "optimize_resource"
        
        # Calculate value based on state
        value = 0.5 + 0.3 * state[0] - 0.2 * state[1]
        
        return logits, value


class RLPlanner:
    """
    Hybrid RL Planner as specified in rl_loop_spec.md.
    """
    
    def __init__(self, 
                 config: RLConfig = RLConfig(),
                 latent_dim: int = 256,
                 action_space_size: int = 32,
                 use_dummy_models: bool = True):
        """
        Initialize the RL Planner.
        
        Args:
            config: RL configuration
            latent_dim: Dimension of the latent state
            action_space_size: Number of possible actions
            use_dummy_models: Whether to use dummy models for testing
        """
        self.config = config
        self.latent_dim = latent_dim
        self.action_space_size = action_space_size
        self.use_dummy_models = use_dummy_models
        
        # Initialize models
        if use_dummy_models:
            self.policy = DummyPolicyNetwork(action_space_size)
            self.forward_model = DummyForwardModel(latent_dim)
        else:
            self.policy = PolicyNetwork(latent_dim, action_space_size)
            # TODO: Use actual forward world model - open issue in forward_world_model.md
            self.forward_model = DummyForwardModel(latent_dim)
        
        # Initialize MCTS
        self.mcts = MCTS(
            config.mcts_config,
            self._forward_model_wrapper,
            self._contradiction_scorer,
            action_space_size
        )
        
        # Training statistics
        self.train_step = 0
        self.episode_rewards = []
    
    def _forward_model_wrapper(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """
        Wrapper for the forward model to handle different input formats.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Tuple of next state and reward
        """
        return self.forward_model(state, action)
    
    def _contradiction_scorer(self, state: np.ndarray) -> float:
        """
        Calculate contradiction loss for a state.
        
        Args:
            state: Current state
            
        Returns:
            Contradiction loss
        """
        # In a real implementation, this would call the Contradiction Engine
        # For now, use a simple heuristic based on state values
        contradiction_level = -state[0]  # First dimension tracks contradiction
        return contradiction_level
    
    def select_action(self, z_t: np.ndarray, use_mcts: bool = True) -> Dict[str, Any]:
        """
        Select an action for a given state.
        
        Args:
            z_t: Latent state representation
            use_mcts: Whether to use MCTS or direct policy
            
        Returns:
            Dictionary containing action details
        """
        if use_mcts:
            # Run MCTS to get improved policy
            action_probs = self.mcts.run(z_t, self.policy)
            
            # Select action with highest probability
            action = max(action_probs, key=action_probs.get)
            
            # Map numeric action to meaningful action type
            action_type = self._map_action_to_type(action)
            
            return {
                "action": action,
                "action_type": action_type,
                "action_probs": action_probs,
                "method": "mcts"
            }
        else:
            # Use direct policy
            if self.use_dummy_models:
                logits, _ = self.policy(z_t)
                action_probs = softmax(logits)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(z_t).unsqueeze(0)
                    logits, _ = self.policy(state_tensor)
                    action_probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
            
            # Select action with highest probability
            action = np.argmax(action_probs)
            
            # Map numeric action to meaningful action type
            action_type = self._map_action_to_type(action)
            
            return {
                "action": int(action),
                "action_type": action_type,
                "action_probs": {i: p for i, p in enumerate(action_probs)},
                "method": "policy"
            }
    
    def _map_action_to_type(self, action: int) -> str:
        """
        Map numeric action to action type string.
        
        Args:
            action: Numeric action ID
            
        Returns:
            Action type string
        """
        # Simple mapping based on action ID modulo 4
        action_map = {
            0: "redistribute_resources",
            1: "optimize_resource",
            2: "mediate",
            3: "pause"
        }
        return action_map.get(action % 4, "unknown")
    
    def should_use_mcts(self, z_t: np.ndarray) -> bool:
        """
        Decide whether to use MCTS based on state characteristics.
        
        Args:
            z_t: Latent state representation
            
        Returns:
            True if MCTS should be used, False otherwise
        """
        # Use MCTS when:
        # 1. Contradiction level is high
        contradiction_level = self._contradiction_scorer(z_t)
        
        # 2. When we're uncertain
        uncertainty = abs(z_t[0]) < 0.2  # Low confidence in contradiction direction
        
        return contradiction_level > 0.3 or uncertainty
    
    def plan(self, z_t: np.ndarray, goal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a plan for a goal from a given state.
        
        Args:
            z_t: Current latent state
            goal: Goal specification
            
        Returns:
            Plan details
        """
        # Decide whether to use MCTS based on state
        use_mcts = self.should_use_mcts(z_t)
        
        # Select action
        action_result = self.select_action(z_t, use_mcts=use_mcts)
        
        # Create plan with additional details
        plan = {
            "chosen_action": action_result["action"],
            "action_type": action_result["action_type"],
            "planning_method": action_result["method"],
            "contradiction_level": self._contradiction_scorer(z_t),
            "goal": goal
        }
        
        # Add predicted outcome
        next_state, reward = self.forward_model(z_t, action_result["action"])
        plan["predicted_next_state"] = next_state.tolist()
        plan["predicted_reward"] = float(reward)
        
        return plan
    
    def update_from_feedback(self, z_t: np.ndarray, action: int, reward: float, next_z_t: np.ndarray) -> None:
        """
        Update policy from feedback (for learning).
        
        Args:
            z_t: Current state
            action: Action taken
            reward: Reward received
            next_z_t: Next state
        """
        # In a real implementation, this would store experiences for batch updates
        # For the prototype, we just log the feedback
        self.episode_rewards.append(reward)
        self.train_step += 1
        
        logger.info(f"Received feedback - action: {action}, reward: {reward:.4f}, step: {self.train_step}")


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax values for a vector.
    
    Args:
        x: Input vector
        
    Returns:
        Softmax probabilities
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()