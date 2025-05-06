"""
Scenario Loader for SAAF-OS
Parses scenario_deck.md and loads scenario definitions.
Implements scenario loading as described in testing_harness.md and scenario_deck.md.
"""
import os
import re
from typing import Dict, Any, List, Optional

SCENARIO_DECK_PATH = os.path.join(os.path.dirname(__file__), '../scenario_deck.md')

class ScenarioLoader:
    """
    Loads and parses scenarios from scenario_deck.md.
    """
    def __init__(self, deck_path: Optional[str] = None):
        self.deck_path = deck_path or SCENARIO_DECK_PATH
        self._scenarios = self._parse_scenarios()

    def _parse_scenarios(self) -> Dict[str, Dict[str, Any]]:
        scenarios = {}
        if not os.path.exists(self.deck_path):
            return scenarios
        with open(self.deck_path) as f:
            content = f.read()
        # Simple parser: each scenario starts with '### Scenario N:'
        scenario_blocks = re.split(r'###\s+Scenario\s+(\d+):', content)
        # Each pair: [<prefix>, num1, block1, num2, block2, ...]
        for i in range(1, len(scenario_blocks), 2):
            num = scenario_blocks[i].strip()
            raw = scenario_blocks[i+1].strip()
            title_match = re.match(r'"([^"]+)"', raw)
            title = title_match.group(1) if title_match else f"Scenario {num}"
            scenarios[num] = {
                "number": num,
                "title": title,
                "raw": raw
            }
        return scenarios

    def list_scenarios(self) -> List[Dict[str, Any]]:
        return [v for v in self._scenarios.values()]

    def get_scenario(self, num_or_title: str) -> Optional[Dict[str, Any]]:
        # Try by number
        if num_or_title in self._scenarios:
            return self._scenarios[num_or_title]
        # Try by title
        for s in self._scenarios.values():
            if s["title"].lower() == num_or_title.lower():
                return s
        return None

# Explicit support for Scenario 3: "Misaligned Labor Plan"
def get_misaligned_labor_plan():
    loader = ScenarioLoader()
    return loader.get_scenario("3")

# ... Unit tests will be implemented in tests/scenarios/test_scenario_loader.py ...
