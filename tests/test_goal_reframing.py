"""
Test suite for the Dialectical Goal Reframing functionality in the Meta-Reasoner module.

This test verifies that the goal reframing functionality correctly handles persistent contradictions
by reframing goals using various dialectical strategies:
1. Reducing precision of goals (converting specific values to flexible ranges)
2. Down-prioritizing subgoals with unresolved tensions
3. Shifting from antagonistic to cooperative language in the approach
"""

import unittest
import numpy as np
import copy
import sys
import os
from typing import List, Dict

# Add the root directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.meta.meta_reasoner import MetaReasoner


class TestGoalReframing(unittest.TestCase):
    """Tests for the Dialectical Goal Reframing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.meta_reasoner = MetaReasoner()
        
        # Create a mock world state
        self.world_state = {
            'solar': 50,
            'battery': 70,
            'crops': {
                'water_level': 45,
                'health': 80
            },
            'time': 12.0
        }
        
        # Create a mock latent state vector (placeholder)
        self.latent_state = np.random.random(32)
        
        # Create a test goal with potential contradictions
        self.goal = {
            'title': 'Optimize Farm Operations',
            'objectives': [
                'maximize crop yield',
                'minimize water usage',
                'ensure optimal solar panel output'
            ],
            'target_state': {
                'water_level': 50,
                'battery': 90,
                'crop_health': 95
            },
            'approach': 'compete against neighboring farms for water resources',
            'subgoals': [
                {
                    'id': 1,
                    'description': 'Water the crops immediately',
                    'priority': 'high'
                },
                {
                    'id': 2,
                    'description': 'Reserve battery power for overnight operations',
                    'priority': 'medium'
                },
                {
                    'id': 3,
                    'description': 'Clean solar panels to increase energy capture',
                    'priority': 'high'
                }
            ]
        }
        
        # Create test history entries to simulate persistent contradictions
        self.meta_reasoner.history = [
            {
                'scenario_id': 'test_scenario',
                'timestamp': np.datetime64('now'),
                'latent_state_hash': 12345,
                'contradiction_before': 0.7,
                'contradiction_after': 0.6,
                'contradiction_reduction': 0.1,
                'rsi_attempted': True,
                'rsi_accepted': False,
                'plan': {
                    'steps': [
                        {'objective': 'Water the crops immediately', 'tension': 0.7},
                        {'objective': 'Clean solar panels', 'tension': 0.2}
                    ]
                }
            },
            {
                'scenario_id': 'test_scenario',
                'timestamp': np.datetime64('now'),
                'latent_state_hash': 12346,
                'contradiction_before': 0.8,
                'contradiction_after': 0.6,
                'contradiction_reduction': 0.2,
                'rsi_attempted': True,
                'rsi_accepted': False,
                'plan': {
                    'steps': [
                        {'objective': 'Water the crops immediately', 'tension': 0.8},
                        {'objective': 'Clean solar panels', 'tension': 0.3}
                    ]
                }
            },
            {
                'scenario_id': 'test_scenario',
                'timestamp': np.datetime64('now'),
                'latent_state_hash': 12347,
                'contradiction_before': 0.7,
                'contradiction_after': 0.6,
                'contradiction_reduction': 0.1,
                'rsi_attempted': True,
                'rsi_accepted': False,
                'plan': {
                    'steps': [
                        {'objective': 'Water the crops immediately', 'tension': 0.7},
                        {'objective': 'Clean solar panels', 'tension': 0.5}
                    ]
                }
            }
        ]
        
        # Create a contradiction history for testing
        self.contradiction_history = [0.7, 0.6, 0.6]

    def test_no_reframing_for_low_contradictions(self):
        """Test that no reframing happens when contradictions are low."""
        low_contradiction_history = [0.2, 0.3, 0.1]
        
        result = self.meta_reasoner.reframe_goal(
            self.world_state, 
            self.latent_state,
            self.goal,
            low_contradiction_history
        )
        
        self.assertIsNone(result, "Goal should not be reframed when contradictions are low")
    
    def test_reframing_for_persistent_contradictions(self):
        """Test that goal is reframed when contradictions persist."""
        result = self.meta_reasoner.reframe_goal(
            self.world_state, 
            self.latent_state,
            self.goal,
            self.contradiction_history
        )
        
        self.assertIsNotNone(result, "Goal should be reframed when contradictions persist")
        self.assertNotEqual(result, self.goal, "Reframed goal should be different from original goal")
    
    def test_goal_precision_reduction(self):
        """Test that numeric goals are converted to flexible ranges."""
        result = self.meta_reasoner.reframe_goal(
            self.world_state, 
            self.latent_state,
            self.goal,
            self.contradiction_history
        )
        
        self.assertIsNotNone(result)
        self.assertIn('constraints', result)
        self.assertIn('ranges', result['constraints'])
        
        # Check that at least some target values have been converted to ranges
        self.assertGreater(len(result['constraints']['ranges']), 0)
        
        # Check that preferences are maintained
        self.assertIn('preferences', result)
    
    def test_antagonistic_language_shift(self):
        """Test that antagonistic language is reframed to cooperative language."""
        # Create a goal with antagonistic language
        antagonistic_goal = copy.deepcopy(self.goal)
        antagonistic_goal['approach'] = "compete against neighboring farms for water resources"
        
        result = self.meta_reasoner.reframe_goal(
            self.world_state, 
            self.latent_state,
            antagonistic_goal,
            self.contradiction_history
        )
        
        self.assertIsNotNone(result)
        self.assertIn('approach', result)
        self.assertNotIn('compete', result['approach'])
        self.assertIn('collaborate', result['approach'])
    
    def test_subgoal_priority_adjustment(self):
        """Test that high-tension subgoals are down-prioritized."""
        result = self.meta_reasoner.reframe_goal(
            self.world_state, 
            self.latent_state,
            self.goal,
            self.contradiction_history
        )
        
        self.assertIsNotNone(result)
        
        # Find the "Water the crops" subgoal which should now be low priority
        water_crops_subgoal = None
        for subgoal in result['subgoals']:
            if 'Water the crops' in subgoal['description']:
                water_crops_subgoal = subgoal
                break
        
        self.assertIsNotNone(water_crops_subgoal)
        self.assertIn('priority', water_crops_subgoal)
        self.assertIn(water_crops_subgoal['priority'], ['low', 'optional'])
        
        # Check that metadata explains the reframing
        self.assertIn('metadata', water_crops_subgoal)
        self.assertIn('reframed', water_crops_subgoal['metadata'])
        self.assertTrue(water_crops_subgoal['metadata']['reframed'])
    
    def test_objective_language_softening(self):
        """Test that absolute terms in objectives are softened."""
        result = self.meta_reasoner.reframe_goal(
            self.world_state, 
            self.latent_state,
            self.goal,
            self.contradiction_history
        )
        
        self.assertIsNotNone(result)
        
        found_softened = False
        for objective in result['objectives']:
            # Check if maximize/minimize terms were softened
            if 'increase' in objective or 'reduce' in objective:
                found_softened = True
                break
        
        self.assertTrue(found_softened, "Objective language should be softened")
        
        # Also check for softening of "ensure" to "improve"
        found_softened = False
        for objective in result['objectives']:
            if 'improve' in objective:
                found_softened = True
                break
        
        self.assertTrue(found_softened, "Objective language should replace 'ensure' with 'improve'")


if __name__ == '__main__':
    unittest.main()