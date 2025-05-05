
# base_agent.py

"""
Abstract base class for all SAAF‑OS agents. Handles message routing, internal state,
ULS encoding, contradiction tracking, and integration with governance and RSI.
"""

import abc
import uuid
import datetime
from typing import Any, Dict, List

class BaseAgent(abc.ABC):
    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.state = {}
        self.z_t = None
        self.message_log: List[Dict[str, Any]] = []
        self.last_tick_time = None
        self.health_status = "OK"
        self.module_type = "base_agent"

    @abc.abstractmethod
    def encode_state(self, u_t: Dict[str, Any]) -> Any:
        """Encode the input state into the Unified Latent Space z_t."""
        pass

    @abc.abstractmethod
    def act(self, z_t: Any) -> Dict[str, Any]:
        """Decide an action based on the current latent state z_t."""
        pass

    @abc.abstractmethod
    def handle_message(self, message: Dict[str, Any]) -> None:
        """Handle an incoming message from the message bus."""
        pass

    def tick(self, u_t: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one agent cycle:
        - Encode state into z_t
        - Act based on z_t
        - Return action
        """
        self.last_tick_time = datetime.datetime.utcnow().isoformat()
        self.z_t = self.encode_state(u_t)
        action = self.act(self.z_t)
        self.log_message({
            "timestamp": self.last_tick_time,
            "type": "agent.tick",
            "z_t": self.z_t,
            "action": action
        })
        return action

    def log_message(self, message: Dict[str, Any]):
        self.message_log.append(message)

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "health": self.health_status,
            "tick_time": self.last_tick_time,
            "z_t": self.z_t
        }

    def receive_task(self, task: Dict[str, Any]) -> None:
        """
        Default task handler — can be overridden for specialized worker roles.
        """
        self.handle_message({
            "type": "task.received",
            "data": task
        })

    def shutdown(self):
        """Clean shutdown hook."""
        self.health_status = "SHUTDOWN"
