"""
Forward World Model (FWM) for SAAF-OS

This module implements the Forward World Model as specified in forward_world_model.md.
It provides differentiable simulations for planning, counterfactual reasoning, and risk analysis
across physical and socio-technical domains.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ForwardWorldModel")

@dataclass
class FWMConfig:
    """Configuration for ForwardWorldModel training and simulation."""
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    num_training_iterations: int = 1000
    batch_size: int = 32
    latent_dim: int = 256
    action_dim: int = 32

@dataclass
class State:
    """Class representing a state in the FWM."""
    vector: np.ndarray

    def __post_init__(self):
        """Ensure vector is a numpy array."""
        if isinstance(self.vector, torch.Tensor):
            self.vector = self.vector.detach().cpu().numpy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {"vector": self.vector.tolist()}

    def distance(self, other: 'State') -> float:
        return np.linalg.norm(self.vector - other.vector)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return np.allclose(self.vector, other.vector)

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.vector, dtype=torch.float32)

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> 'State':
        return State(tensor.detach().cpu().numpy())

@dataclass
class Action:
    """Class representing an action in the FWM."""
    vector: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary for serialization."""
        return {"vector": self.vector.tolist()}

    @staticmethod
    def continuous(values: List[float]) -> 'Action':
        return Action(np.array(values))

    @staticmethod
    def discrete(index: int, action_space_size: int) -> 'Action':
        vec = np.zeros(action_space_size)
        vec[index] = 1
        return Action(vec)

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.vector, dtype=torch.float32)

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> 'Action':
        return Action(tensor.detach().cpu().numpy())

@dataclass
class Trajectory:
    """Class representing a trajectory (sequence of states and actions)."""
    states: List[State]
    actions: List[Action]
    rewards: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary for serialization."""
        return {
            "states": [s.to_dict() for s in self.states],
            "actions": [a.to_dict() for a in self.actions],
            "rewards": self.rewards
        }

    def add_transition(self, state: State, action: Action, reward: float, next_state: State):
        if not self.states:
            self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states.append(next_state)

    def cumulative_reward(self) -> float:
        return sum(self.rewards)

    def discounted_reward(self, gamma: float = 0.9) -> float:
        return sum(r * (gamma ** i) for i, r in enumerate(self.rewards))

    def to_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        X = []
        y = []
        for i in range(len(self.actions)):
            X.append(np.concatenate([self.states[i].vector, self.actions[i].vector]))
            y.append(self.states[i+1].vector)
        return np.array(X), np.array(y)

class FWMDynamicsModel(nn.Module):
    """Neural dynamics model for the Forward World Model."""
    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return self.network(x)

class ForwardWorldModel:
    """Main class implementing the Forward World Model."""
    def __init__(self, state_dim: int, action_dim: int, config: Optional[FWMConfig] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or FWMConfig()
        self.model = FWMDynamicsModel(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.MSELoss()

    def save(self, filename: str):
        torch.save({'state_dict': self.model.state_dict()}, filename)

    def load(self, path: str):
        try:
            checkpoint = torch.load(path)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                logger.info(f"Loaded model state_dict from {path}")
            else:
                logger.warning(f"No 'state_dict' found in checkpoint at {path}. Model not loaded.")
        except FileNotFoundError:
            logger.error(f"Model file not found at {path}. Model not loaded.")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}. Model may not be fully loaded.")

    def predict_next_state(self, state: State, action: Action) -> State:
        self.model.eval()
        with torch.no_grad():
            state_tensor = state.to_tensor().unsqueeze(0)
            action_tensor = action.to_tensor().unsqueeze(0)
            next_state_tensor = self.model(state_tensor, action_tensor)
            return State.from_tensor(next_state_tensor.squeeze(0))

    def simulate_trajectory(self, initial_state: State, actions: List[Action], reward_function) -> Trajectory:
        states = [initial_state]
        rewards = []
        for action in actions:
            next_state = self.predict_next_state(states[-1], action)
            reward = reward_function(states[-1], action, next_state)
            states.append(next_state)
            rewards.append(reward)
        return Trajectory(states, actions, rewards)

    def train(self, X: np.ndarray, y: np.ndarray, num_iterations: Optional[int] = None):
        self.model.train()
        iterations = num_iterations or self.config.num_training_iterations
        for iteration in range(iterations):
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            
            state_tensors = X_tensor[:, :self.state_dim]
            action_tensors = X_tensor[:, self.state_dim:]
            
            predictions = self.model(state_tensors, action_tensors)
            loss = self.loss_fn(predictions, y_tensor)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if iteration % 100 == 0:
                logger.info(f"Training: Iteration {iteration}, Loss: {loss.item()}")
        logger.info("Model training complete.")

    def evaluate(self, test_trajectories: List[Trajectory]) -> float:
        self.model.eval()
        total_mse = 0.0
        count = 0
        with torch.no_grad():
            for trajectory in test_trajectories:
                X, y_true = trajectory.to_dataset()
                if X.shape[0] > 0:
                    y_pred_tensors = self.model(torch.tensor(X[:, :self.state_dim], dtype=torch.float32), torch.tensor(X[:, self.state_dim:], dtype=torch.float32))
                    y_pred = y_pred_tensors.numpy()
                    total_mse += np.mean((y_true - y_pred)**2)
                    count += 1
        return total_mse / count if count > 0 else 0.0

    @classmethod
    def from_config(cls, state_dim: int, action_dim: int, config: FWMConfig) -> 'ForwardWorldModel':
        return cls(state_dim, action_dim, config)

    def prediction_error_metrics(self, actual: State, predicted: State) -> Tuple[float, float, float]:
        diff = actual.vector - predicted.vector
        mae = np.mean(np.abs(diff))
        mse = np.mean(diff ** 2)
        rmse = np.sqrt(mse)
        return float(mae), float(mse), float(rmse)
