import os
import numpy as np
import time
import json
from modules.contradiction.engine import ContradictionEngine
from modules.contradiction.bus import contradiction_bus
from modules.goals.delegator import GoalDelegator
from modules.world_model.fwm import ForwardWorldModel
from modules.planning.rl_planner import RLPlanner, RLConfig

AGENT_IDS = [f'agent_{i+1}' for i in range(3)]

class SimAgent:
    def __init__(self, agent_id, world_state, goal_queue):
        self.agent_id = agent_id
        self.latent_state = np.random.uniform(0, 1, 16)
        self.planner = RLPlanner(RLConfig(latent_dim=16), use_dummy_models=True)
        self.goal_queue = goal_queue
        self.episodes = []
        self.world_state = world_state
        self.contradiction_engine = ContradictionEngine()
        contradiction_bus.subscribe(self.handle_broadcast)
        self.log_path = f'memory/{agent_id}_episodes.jsonl'

    def handle_broadcast(self, event):
        # Update local contradiction graph or trigger synthesis
        if event['agent_id'] != self.agent_id:
            self.contradiction_engine.contradiction_graph.add_node(event['c_type'])  # Simplified
            # Could trigger synthesis proposal here

    def log_episode(self, record):
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def step(self):
        if not self.goal_queue:
            return False
        goal = self.goal_queue.pop(0)
        plan = self.planner.plan(z_t=self.latent_state, goal=goal)
        # Simulate contradiction detection
        contradiction_score = np.random.rand()
        event = {
            'agent_id': self.agent_id,
            'c_type': goal.get('contradiction_type', 'generic'),
            'z_t': self.latent_state.tolist(),
            'timestamp': time.time()
        }
        contradiction_bus.broadcast(event)
        # Log episode
        record = {
            'goal': goal,
            'plan': plan,
            'contradiction_score': contradiction_score,
            'z_t': self.latent_state.tolist(),
            'timestamp': time.time()
        }
        self.log_episode(record)
        return True

def main():
    # Shared world state and contradiction graph
    world_state = {'resources': {'solar': 100, 'water': 50}}
    goals = [
        {'description': 'Harvest field', 'value_vector': [1,0,0,0], 'contradiction_type': 'resource_conflict'},
        {'description': 'Irrigate crops', 'value_vector': [0,1,0,0], 'contradiction_type': 'resource_conflict'},
        {'description': 'Repair bot', 'value_vector': [0,0,1,0], 'contradiction_type': 'maintenance'},
    ]
    # Each agent gets a copy of the goal queue
    agents = [SimAgent(agent_id, world_state, goals.copy()) for agent_id in AGENT_IDS]
    delegator = GoalDelegator(AGENT_IDS)
    agent_states = {a.agent_id: {'value_vector': a.latent_state.tolist()} for a in agents}
    # Main loop
    for step in range(5):
        for agent in agents:
            if agent.goal_queue:
                # Inter-agent delegation
                if np.random.rand() < 0.3:
                    goal = agent.goal_queue.pop(0)
                    delegate_to = delegator.delegate(goal, agent_states)
                    if delegate_to != agent.agent_id:
                        # Actually transfer goal
                        for a in agents:
                            if a.agent_id == delegate_to:
                                a.goal_queue.append(goal)
                                break
                        continue
                    else:
                        agent.goal_queue.insert(0, goal)
                agent.step()
    # Archive all agent logs
    import zipfile
    ts = int(time.time())
    zip_path = f'memory/agents_logs_{ts}.zip'
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for agent in agents:
            if os.path.exists(agent.log_path):
                zipf.write(agent.log_path, os.path.basename(agent.log_path))
    print(f'Agent logs archived to {zip_path}')

if __name__ == '__main__':
    main()
