"""
Unified Latent Space (ULS) Encoder for SAAF-OS

This module implements the ULS Encoder as specified in latent_encoding_spec.md.
It converts raw system state (u_t) into the unified latent representation (z_t)
that serves as the common substrate for all SAAF-OS modules.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Any, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ULSEncoder")


class StateEncoder(nn.Module):
    """
    Neural network encoder for converting input state dictionaries into latent vectors.
    """
    
    def __init__(self, output_dim: int = 256):
        """
        Initialize the encoder network.
        
        Args:
            output_dim: Dimension of the output latent space
        """
        super().__init__()
        
        # Encode modules/categories
        self.robotics_encoder = nn.Sequential(
            nn.Linear(20, 64),  # Assume robotics data is ~20 dimensional
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU()
        )
        
        self.environment_encoder = nn.Sequential(
            nn.Linear(16, 64),  # Assume environment data is ~16 dimensional
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU()
        )
        
        self.census_encoder = nn.Sequential(
            nn.Linear(12, 32),  # Assume census data is ~12 dimensional
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )
        
        self.value_encoder = nn.Sequential(
            nn.Linear(8, 32),  # Assume value vector is ~8 dimensional
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )
        
        self.contradiction_encoder = nn.Sequential(
            nn.Linear(4, 16),  # Assume contradiction data is ~4 dimensional
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU()
        )
        
        # Final integration network
        self.integration_network = nn.Sequential(
            nn.Linear(64 + 64 + 32 + 32 + 16, 128),  # Sum of all encoders
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Output dimension
        self.output_dim = output_dim
    
    def forward(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            state: Dictionary with state information
            
        Returns:
            Latent space encoding of the state
        """
        # Extract and encode each aspect of the state
        # Use zeros as default for missing parts
        
        # Robotics
        if "robotics" in state:
            # Convert the robotics dict to a tensor
            robotics_data = []
            if "energy_usage" in state["robotics"]:
                robotics_data.append(state["robotics"]["energy_usage"])
            if "joint_positions" in state["robotics"]:
                positions = self._to_tensor(state["robotics"]["joint_positions"])
                robotics_data.append(positions.reshape(-1))
            if "tool_status" in state["robotics"]:
                tool_status = self._to_tensor(state["robotics"]["tool_status"])
                robotics_data.append(tool_status.reshape(-1))
            
            # Concatenate and pad if necessary
            robotics_tensor = torch.cat([d.reshape(-1) for d in robotics_data])
            if robotics_tensor.shape[0] < 20:
                robotics_tensor = F.pad(robotics_tensor, (0, 20 - robotics_tensor.shape[0]))
            elif robotics_tensor.shape[0] > 20:
                robotics_tensor = robotics_tensor[:20]
            
            robotics_encoding = self.robotics_encoder(robotics_tensor)
        else:
            robotics_encoding = torch.zeros(64)
        
        # Environment
        if "environment_map" in state:
            # Convert the environment dict to a tensor
            env_data = []
            for k, v in state["environment_map"].items():
                env_data.append(self._to_tensor(v).reshape(-1))
            
            # Concatenate and pad if necessary
            env_tensor = torch.cat(env_data) if env_data else torch.zeros(1)
            if env_tensor.shape[0] < 16:
                env_tensor = F.pad(env_tensor, (0, 16 - env_tensor.shape[0]))
            elif env_tensor.shape[0] > 16:
                env_tensor = env_tensor[:16]
            
            environment_encoding = self.environment_encoder(env_tensor)
        else:
            environment_encoding = torch.zeros(64)
        
        # Census
        if "census" in state:
            # Convert the census dict to a tensor
            census_data = []
            for k, v in state["census"].items():
                census_data.append(self._to_tensor(v).reshape(-1))
            
            # Concatenate and pad if necessary
            census_tensor = torch.cat(census_data) if census_data else torch.zeros(1)
            if census_tensor.shape[0] < 12:
                census_tensor = F.pad(census_tensor, (0, 12 - census_tensor.shape[0]))
            elif census_tensor.shape[0] > 12:
                census_tensor = census_tensor[:12]
            
            census_encoding = self.census_encoder(census_tensor)
        else:
            census_encoding = torch.zeros(32)
        
        # Value Vector
        if "value_vector" in state:
            # Convert the value vector to a tensor
            value_data = []
            for k, v in state["value_vector"].items():
                value_data.append(torch.tensor(float(v)).reshape(-1))
            
            # Concatenate and pad if necessary
            value_tensor = torch.cat(value_data) if value_data else torch.zeros(1)
            if value_tensor.shape[0] < 8:
                value_tensor = F.pad(value_tensor, (0, 8 - value_tensor.shape[0]))
            elif value_tensor.shape[0] > 8:
                value_tensor = value_tensor[:8]
            
            value_encoding = self.value_encoder(value_tensor)
        else:
            value_encoding = torch.zeros(32)
        
        # Contradiction
        if "contradictions" in state:
            # Convert the contradiction dict to a tensor
            contradiction_data = []
            if "tension" in state["contradictions"]:
                contradiction_data.append(torch.tensor(float(state["contradictions"]["tension"])).reshape(-1))
            if "type" in state["contradictions"]:
                # One-hot encode the type
                contradiction_types = ["resource", "value", "goal", "energy_constraint", "unknown"]
                c_type = state["contradictions"]["type"]
                c_type_idx = contradiction_types.index(c_type) if c_type in contradiction_types else 4
                c_type_tensor = F.one_hot(torch.tensor(c_type_idx), len(contradiction_types))
                contradiction_data.append(c_type_tensor.float())
            
            # Concatenate and pad if necessary
            contradiction_tensor = torch.cat(contradiction_data) if contradiction_data else torch.zeros(1)
            if contradiction_tensor.shape[0] < 4:
                contradiction_tensor = F.pad(contradiction_tensor, (0, 4 - contradiction_tensor.shape[0]))
            elif contradiction_tensor.shape[0] > 4:
                contradiction_tensor = contradiction_tensor[:4]
            
            contradiction_encoding = self.contradiction_encoder(contradiction_tensor)
        else:
            contradiction_encoding = torch.zeros(16)
        
        # Integrate all encodings
        combined_encoding = torch.cat([
            robotics_encoding,
            environment_encoding,
            census_encoding,
            value_encoding,
            contradiction_encoding
        ], dim=0)
        
        # Final integration
        z_t = self.integration_network(combined_encoding)
        
        # Normalize to unit sphere (as specified in latent_encoding_spec.md)
        z_t = F.normalize(z_t, p=2, dim=0)
        
        return z_t
    
    def _to_tensor(self, value: Union[float, List, np.ndarray]) -> torch.Tensor:
        """
        Convert a value to a PyTorch tensor.
        
        Args:
            value: Value to convert
            
        Returns:
            PyTorch tensor
        """
        if isinstance(value, (float, int)):
            return torch.tensor(float(value)).reshape(1)
        elif isinstance(value, list):
            return torch.tensor(value, dtype=torch.float)
        elif isinstance(value, np.ndarray):
            return torch.from_numpy(value.astype(np.float32))
        elif isinstance(value, torch.Tensor):
            return value.float()
        else:
            return torch.tensor(0.0).reshape(1)


class ULSEncoder:
    """
    Main class implementing the ULS Encoder as specified in latent_encoding_spec.md.
    """
    
    def __init__(self, output_dim: int = 256, use_neural_encoder: bool = True):
        """
        Initialize the ULS Encoder.
        
        Args:
            output_dim: Dimension of the latent space
            use_neural_encoder: Whether to use neural network encoder
        """
        self.output_dim = output_dim
        self.use_neural_encoder = use_neural_encoder
        
        if use_neural_encoder:
            self.encoder = StateEncoder(output_dim)
        else:
            self.encoder = None
    
    def encode_state(self, u_t: Dict[str, Any]) -> np.ndarray:
        """
        Encode raw system state (u_t) into the unified latent space (z_t).
        
        Args:
            u_t: Raw system state dictionary
            
        Returns:
            Latent space representation (z_t)
        """
        if self.use_neural_encoder:
            # Use neural encoder
            with torch.no_grad():
                z_t = self.encoder(u_t).detach().numpy()
            return z_t
        else:
            # Use simple encoding for testing
            return self._simple_encode(u_t)
    
    def _simple_encode(self, u_t: Dict[str, Any]) -> np.ndarray:
        """
        Simple encoding function for testing.
        
        Args:
            u_t: Raw system state dictionary
            
        Returns:
            Latent space representation (z_t)
        """
        # Initialize random latent vector
        z_t = np.random.randn(self.output_dim)
        
        # Modify first dimensions based on important state variables
        if "robotics" in u_t and "energy_usage" in u_t["robotics"]:
            # Higher energy usage -> lower first dimension (more contradiction)
            z_t[0] = -u_t["robotics"]["energy_usage"]
        
        if "value_vector" in u_t and "energy_efficiency" in u_t["value_vector"]:
            # Higher efficiency value -> better energy management
            z_t[1] = -u_t["value_vector"]["energy_efficiency"]
        
        if "contradictions" in u_t and "tension" in u_t["contradictions"]:
            # Higher tension -> lower first dimension (more contradiction)
            z_t[0] = -u_t["contradictions"]["tension"]
        
        if "census" in u_t and "bot_distribution" in u_t["census"]:
            # Bot distribution affects resource allocation
            dist = u_t["census"]["bot_distribution"]
            if isinstance(dist, np.ndarray) and len(dist) > 0:
                z_t[2] = np.std(dist) * 0.5  # Higher standard deviation -> more uneven distribution
        
        # Normalize to unit sphere (as specified in latent_encoding_spec.md)
        z_t = z_t / np.linalg.norm(z_t)
        
        return z_t
    
    def decode_state(self, z_t: np.ndarray) -> Dict[str, float]:
        """
        Decode latent space representation (z_t) into interpretable metrics.
        This is a simplified version that extracts key metrics from the latent space.
        
        Args:
            z_t: Latent space representation
            
        Returns:
            Dictionary of interpretable metrics
        """
        # Simple linear projection for interpretable metrics
        metrics = {
            "contradiction_level": float(-z_t[0]),  # First dimension tracks contradiction
            "energy_efficiency": float(-z_t[1]),    # Second dimension tracks energy usage
            "resource_distribution": float(z_t[2]), # Third dimension tracks resource distribution
            "value_alignment": float(z_t[3])        # Fourth dimension tracks value alignment
        }
        
        return metrics

    def train(self, dataset: List[Tuple[Dict[str, Any], np.ndarray]]) -> float:
        """
        Train the encoder using a dataset of (u_t, z_t) pairs.
        
        Args:
            dataset: List of (u_t, z_t) pairs
            
        Returns:
            Training loss
        """
        if not self.use_neural_encoder or self.encoder is None:
            logger.error("Cannot train without a neural encoder")
            return float('inf')
        
        # Training setup
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()
        
        # Training loop (simplified)
        total_loss = 0.0
        
        # Convert dataset
        u_ts = [u_t for u_t, _ in dataset]
        z_ts = [torch.tensor(z_t, dtype=torch.float) for _, z_t in dataset]
        
        # One epoch
        for u_t, target_z_t in zip(u_ts, z_ts):
            # Forward pass
            predicted_z_t = self.encoder(u_t)
            
            # Compute loss
            loss = loss_fn(predicted_z_t, target_z_t)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset) if dataset else 0
        logger.info(f"Training completed with average loss: {avg_loss:.4f}")
        return avg_loss


class DummyULSEncoder(ULSEncoder):
    """
    Dummy implementation of the ULS Encoder for testing.
    Uses the simple encoding from the parent class.
    """
    
    def __init__(self, output_dim: int = 256):
        """
        Initialize the dummy ULS Encoder.
        
        Args:
            output_dim: Dimension of the latent space
        """
        super().__init__(output_dim, use_neural_encoder=False)
        logger.info("Initialized Dummy ULS Encoder")