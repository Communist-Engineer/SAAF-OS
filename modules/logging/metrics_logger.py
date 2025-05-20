import json
import os
from datetime import datetime

LOG_PATH = os.path.join('memory', 'logs.jsonl')

def _write_log(entry):
    entry['timestamp'] = datetime.utcnow().isoformat()
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')

def log_tension_score(z_t, c_t):
    _write_log({
        'type': 'tension_score',
        'z_t': z_t,
        'c_t': c_t
    })

def log_value_vector(vector):
    _write_log({
        'type': 'value_vector',
        'vector': vector
    })

def log_reward_trace(reward):
    _write_log({
        'type': 'reward',
        'reward': reward
    })

def log_contradiction_event(event):
    _write_log({
        'type': 'contradiction_event',
        'event': event
    })
