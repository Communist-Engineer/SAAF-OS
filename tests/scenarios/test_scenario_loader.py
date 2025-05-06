"""
Unit tests for ScenarioLoader (simulation/loader.py)
Covers: scenario parsing, listing, and Scenario 3 retrieval.
Implements scenario loading checks per testing_harness.md.
"""
import os
import tempfile
import pytest
from simulation.loader import ScenarioLoader, get_misaligned_labor_plan

SCENARIO_MD = '''
### Scenario 1: "Basic Alignment"
Description: ...

### Scenario 2: "Resource Shortfall"
Description: ...

### Scenario 3: "Misaligned Labor Plan"
Description: This scenario tests contradiction detection and plan correction.
'''

@pytest.fixture(scope="function")
def temp_scenario_deck(monkeypatch):
    tmpfile = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    tmpfile.write(SCENARIO_MD)
    tmpfile.flush()
    monkeypatch.setattr('simulation.loader.SCENARIO_DECK_PATH', tmpfile.name)
    yield tmpfile.name
    tmpfile.close()
    os.unlink(tmpfile.name)

def test_list_scenarios(temp_scenario_deck):
    loader = ScenarioLoader()
    scenarios = loader.list_scenarios()
    assert len(scenarios) == 3
    titles = [s['title'] for s in scenarios]
    assert "Misaligned Labor Plan" in titles

def test_get_scenario_by_number(temp_scenario_deck):
    loader = ScenarioLoader()
    s3 = loader.get_scenario("3")
    assert s3 is not None
    assert s3['title'] == "Misaligned Labor Plan"

def test_get_scenario_by_title(temp_scenario_deck):
    loader = ScenarioLoader()
    s3 = loader.get_scenario("Misaligned Labor Plan")
    assert s3 is not None
    assert s3['number'] == "3"

def test_get_misaligned_labor_plan(temp_scenario_deck):
    s3 = get_misaligned_labor_plan()
    assert s3 is not None
    assert "contradiction" in s3['raw'].lower()
