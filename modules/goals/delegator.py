import numpy as np
from typing import Dict, Any, List

class GoalDelegator:
    """
    Assigns goals to agents based on resource needs, tension, and value alignment.
    Tracks delegation events and success rates.
    """
    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids
        self.delegation_log = []
        self.success_count = 0
        self.total_count = 0

    def estimate_tension(self, goal: Dict[str, Any], agent_state: Dict[str, Any]) -> float:
        # Example: tension is higher if agent's value vector mismatches goal needs
        goal_vec = np.array(goal.get('value_vector', [0]*4))
        agent_vec = np.array(agent_state.get('value_vector', [0]*4))
        return float(np.linalg.norm(goal_vec - agent_vec))

    def best_fit_agent(self, goal: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]]) -> str:
        # Choose agent with lowest tension estimate
        scores = {aid: self.estimate_tension(goal, state) for aid, state in agent_states.items()}
        return min(scores, key=scores.get)

    def delegate(self, goal: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]]) -> str:
        agent_id = self.best_fit_agent(goal, agent_states)
        tension = self.estimate_tension(goal, agent_states[agent_id])
        event = {
            'goal': goal,
            'agent_id': agent_id,
            'tension': tension,
            'timestamp': float(np.round(time.time(), 3))
        }
        self.delegation_log.append(event)
        self.total_count += 1
        return agent_id

    def log_success(self, success: bool):
        if success:
            self.success_count += 1

    def delegation_success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count else 0.0
