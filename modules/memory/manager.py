"""
MemoryManager: Episodic and Semantic Memory for SAAF-OS
Implements memory_spec.md sections 2–4.
"""
import os
import json
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from rdflib import Graph, URIRef, Literal, Namespace

MEMORY_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/memory/')
EPISODIC_FILE = os.path.join(MEMORY_DATA_PATH, 'episodic.jsonl')
PATCH_LOG_FILE = os.path.join(MEMORY_DATA_PATH, 'patch_history.jsonl')
SEMANTIC_FILE = os.path.join(MEMORY_DATA_PATH, 'semantic_triples.ttl')

os.makedirs(MEMORY_DATA_PATH, exist_ok=True)

class MemoryManager:
    """
    Episodic and Semantic Memory Manager for SAAF-OS.
    Implements retrieval endpoints as defined in memory_spec.md.
    """
    def __init__(self):
        self._episodic_lock = threading.Lock()
        self._patch_lock = threading.Lock()
        self._semantic_graph = Graph()
        if os.path.exists(SEMANTIC_FILE):
            self._semantic_graph.parse(SEMANTIC_FILE, format='turtle')

    # Episodic Memory
    def write_episode(self, episode: Dict[str, Any]) -> None:
        """Append an episode to episodic memory."""
        with self._episodic_lock:
            with open(EPISODIC_FILE, 'a') as f:
                f.write(json.dumps(episode) + '\n')

    def read_episodes(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read episodes, optionally filtered by ISO8601 time window."""
        episodes = []
        with self._episodic_lock:
            if not os.path.exists(EPISODIC_FILE):
                return []
            with open(EPISODIC_FILE) as f:
                for line in f:
                    ep = json.loads(line)
                    ts = ep.get('timestamp')
                    if start_time and ts < start_time:
                        continue
                    if end_time and ts > end_time:
                        continue
                    episodes.append(ep)
        return episodes

    # Semantic Memory (RDF triples)
    def add_triple(self, subj: str, pred: str, obj: str) -> None:
        """Add a semantic triple to the store."""
        s = URIRef(subj)
        p = URIRef(pred)
        o = Literal(obj)
        self._semantic_graph.add((s, p, o))
        self._semantic_graph.serialize(destination=SEMANTIC_FILE, format='turtle')

    def query_triples(self, sparql: str) -> List[Dict[str, Any]]:
        """Run a SPARQL query and return results as dicts."""
        results = []
        for row in self._semantic_graph.query(sparql):
            results.append({str(k): str(v) for k, v in row.asdict().items()})
        return results

    # Patch History
    def log_patch(self, patch: Dict[str, Any]) -> None:
        """Append a patch event to the patch history log."""
        with self._patch_lock:
            with open(PATCH_LOG_FILE, 'a') as f:
                f.write(json.dumps(patch) + '\n')

    def get_patch_history(self, patch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return patch history, optionally filtered by patch_id."""
        patches = []
        with self._patch_lock:
            if not os.path.exists(PATCH_LOG_FILE):
                return []
            with open(PATCH_LOG_FILE) as f:
                for line in f:
                    p = json.loads(line)
                    if patch_id and p.get('patch_id') != patch_id:
                        continue
                    patches.append(p)
        return patches

# ... Unit tests will be implemented in tests/test_memory_manager.py ...
