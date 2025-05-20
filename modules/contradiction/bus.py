import json
import os
import time
from typing import Dict, Any

class ContradictionBus:
    """
    Simple in-memory contradiction broadcast system for multi-agent SAAF-OS.
    Agents can log and broadcast contradiction events to all others.
    """
    def __init__(self):
        self.subscribers = []
        self.log_path = os.path.join('diagnostics', 'contradiction_broadcasts.jsonl')
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def broadcast(self, event: Dict[str, Any]):
        event['timestamp'] = time.time()
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + '\n')
        for cb in self.subscribers:
            cb(event)

# Singleton instance for global use
contradiction_bus = ContradictionBus()
