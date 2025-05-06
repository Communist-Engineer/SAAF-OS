"""
Unit tests for MemoryManager (modules/memory/manager.py)
Covers: episodic CRUD, semantic triple store, patch log, time-window query.
Implements memory_spec.md §5.
"""
import os
import shutil
import tempfile
import pytest
from modules.memory.manager import MemoryManager

@pytest.fixture(scope="function")
def temp_memory_dir(monkeypatch):
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setattr('modules.memory.manager.MEMORY_DATA_PATH', tmpdir)
    monkeypatch.setattr('modules.memory.manager.EPISODIC_FILE', os.path.join(tmpdir, 'episodic.jsonl'))
    monkeypatch.setattr('modules.memory.manager.PATCH_LOG_FILE', os.path.join(tmpdir, 'patch_history.jsonl'))
    monkeypatch.setattr('modules.memory.manager.SEMANTIC_FILE', os.path.join(tmpdir, 'semantic_triples.ttl'))
    yield tmpdir
    shutil.rmtree(tmpdir)

def test_episodic_crud(temp_memory_dir):
    mm = MemoryManager()
    ep1 = {"timestamp": "2025-05-05T10:00:00Z", "event": "foo"}
    ep2 = {"timestamp": "2025-05-05T11:00:00Z", "event": "bar"}
    mm.write_episode(ep1)
    mm.write_episode(ep2)
    all_eps = mm.read_episodes()
    assert len(all_eps) == 2
    # Time window query
    eps = mm.read_episodes(start_time="2025-05-05T10:30:00Z")
    assert len(eps) == 1
    assert eps[0]["event"] == "bar"

def test_semantic_triple_store(temp_memory_dir):
    mm = MemoryManager()
    mm.add_triple("http://ex.org/a", "http://ex.org/b", "c")
    q = """
    SELECT ?s ?p ?o WHERE { ?s ?p ?o . FILTER(?o = 'c') }
    """
    results = mm.query_triples(q)
    assert any(r["o"] == "c" for r in results)

def test_patch_history_log(temp_memory_dir):
    mm = MemoryManager()
    patch = {"patch_id": "p1", "desc": "test"}
    mm.log_patch(patch)
    mm.log_patch({"patch_id": "p2", "desc": "other"})
    all_patches = mm.get_patch_history()
    assert len(all_patches) == 2
    filtered = mm.get_patch_history(patch_id="p1")
    assert len(filtered) == 1
    assert filtered[0]["desc"] == "test"
