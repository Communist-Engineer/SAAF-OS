"""
Meta-Reasoner Module for SAAF-OS

This module implements a reflective, system-level controller that oversees the planning
cycle and adapts internal behavior based on contradictions, failure patterns,
and synthesis outcomes.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import copy

logger = logging.getLogger(__name__)

class MetaReasoner:
    """
    MetaReasoner acts as a reflective, system-level controller that oversees the planning cycle
    and adapts internal behavior based on contradictions, failure patterns, and synthesis outcomes.
    """
    
    def __init__(self):
        """Initialize the MetaReasoner with history tracking."""
        self.history = []
        self.contradiction_threshold = 0.3  # Default threshold for high contradiction
        self.rsi_failure_threshold = 2  # Number of consecutive RSI failures to trigger reframing
        self.current_rsi_failures = 0
        self.planner_performance = {
            'retrieved': {'success_rate': 0.5, 'weight': 1.0},
            'distilled': {'success_rate': 0.5, 'weight': 1.0},
            'rl': {'success_rate': 0.5, 'weight': 1.0}
        }
        self.learning_rate = 0.1  # Rate at which to update weights based on new observations
        
    def observe(self, scenario_id: str, z_t: np.ndarray, plan: Dict, result: Dict) -> None:
        """
        Log each planning cycle and contradiction outcome.
        
        Args:
            scenario_id: Identifier for the current scenario
            z_t: Current latent state vector
            plan: The executed plan
            result: Results including contradiction scores and RSI status
        """
        # Extract relevant metrics
        pre_tension = result.get('pre_score', 0.0)
        post_tension = result.get('post_score', 0.0)
        rsi_status = result.get('rsi_patch') is not None
        rsi_accepted = result.get('accepted', False)
        
        # Track RSI failures
        if rsi_status and not rsi_accepted:
            self.current_rsi_failures += 1
        else:
            self.current_rsi_failures = 0
        
        # Store in history
        entry = {
            'scenario_id': scenario_id,
            'timestamp': np.datetime64('now'),
            'latent_state_hash': hash(z_t.tobytes()),
            'plan_energy': plan.get('total_energy', 0.0),
            'contradiction_before': pre_tension,
            'contradiction_after': post_tension,
            'contradiction_reduction': pre_tension - post_tension,
            'rsi_attempted': rsi_status,
            'rsi_accepted': rsi_accepted,
            'planner_used': plan.get('arbitration_method', 'unknown')
        }
        
        self.history.append(entry)
        logger.debug(f"MetaReasoner observed episode: {entry}")
    
    def should_reframe_goal(self, result: Dict) -> bool:
        """
        Return True if contradiction scores remain high after synthesis, or RSI repeatedly fails.
        
        Args:
            result: Results including contradiction scores and RSI status
        
        Returns:
            bool: True if goal reframing is recommended
        """
        # Check if contradiction remains high after synthesis
        post_tension = result.get('post_score', 0.0)
        pre_tension = result.get('pre_score', 0.0)
        
        # Calculate contradiction reduction percentage
        if pre_tension > 0:
            reduction_percentage = (pre_tension - post_tension) / pre_tension
        else:
            reduction_percentage = 0
        
        # Check if reduction is insufficient and contradiction remains high
        insufficient_reduction = reduction_percentage < 0.3 and post_tension > self.contradiction_threshold
        
        # Check for RSI failures
        rsi_failures_exceeded = self.current_rsi_failures >= self.rsi_failure_threshold
        
        # Check recent history for persistent contradictions
        persistent_contradictions = False
        if len(self.history) >= 3:
            recent_contradictions = [entry['contradiction_after'] for entry in self.history[-3:]]
            if all(c > self.contradiction_threshold for c in recent_contradictions):
                persistent_contradictions = True
        
        return insufficient_reduction or rsi_failures_exceeded or persistent_contradictions
    
    def recommend_planner(self, contradiction_score: float) -> str:
        """
        Returns 'retrieved', 'distilled', or 'rl' depending on contradiction score.
        Used to override planner arbitration dynamically.
        
        Args:
            contradiction_score: Current contradiction score
        
        Returns:
            str: Recommended planner strategy
        """
        # Very low contradiction - use retrieved plans from memory
        if contradiction_score < 0.1:
            return 'retrieved'
        # High contradiction - use RL planner for more creative solutions
        elif contradiction_score > 0.5:
            return 'rl'
        # Medium contradiction - use distilled planner for balanced approach
        else:
            return 'distilled'
    
    def get_insights(self) -> Dict[str, Any]:
        """
        Analyze history to extract insights about planner performance.
        
        Returns:
            Dict: Insights about planner performance and contradiction patterns
        """
        if not self.history:
            return {"status": "insufficient_data"}
            
        # Calculate average contradiction reduction by planner type
        planner_stats = {}
        for entry in self.history:
            planner = entry['planner_used']
            if planner not in planner_stats:
                planner_stats[planner] = {
                    'count': 0, 
                    'total_reduction': 0.0,
                    'initial_contradictions': []
                }
            
            planner_stats[planner]['count'] += 1
            planner_stats[planner]['total_reduction'] += entry['contradiction_reduction']
            planner_stats[planner]['initial_contradictions'].append(entry['contradiction_before'])
        
        # Calculate averages
        for planner in planner_stats:
            if planner_stats[planner]['count'] > 0:
                planner_stats[planner]['avg_reduction'] = (
                    planner_stats[planner]['total_reduction'] / planner_stats[planner]['count']
                )
                planner_stats[planner]['avg_initial'] = (
                    sum(planner_stats[planner]['initial_contradictions']) / 
                    len(planner_stats[planner]['initial_contradictions'])
                )
            else:
                planner_stats[planner]['avg_reduction'] = 0
                planner_stats[planner]['avg_initial'] = 0
        
        return {
            "planner_stats": planner_stats,
            "total_episodes": len(self.history),
            "rsi_attempts": sum(1 for entry in self.history if entry['rsi_attempted']),
            "rsi_accepted": sum(1 for entry in self.history if entry['rsi_accepted']),
            "high_contradiction_rate": sum(1 for entry in self.history 
                                          if entry['contradiction_after'] > self.contradiction_threshold) / 
                                     len(self.history) if self.history else 0
        }
    
    def learn_from_outcome(self, plan=None, z_t=None, u_t=None, original_world_state=None, 
                        score=None, execution_success=None,
                        contradiction_before=None, contradictions_before=None, 
                        contradiction_after=None, contradictions_after=None, **kwargs) -> None:
        """
        Update internal models based on the outcome of a plan execution.
        This creates a feedback loop where the MetaReasoner improves over time.
        
        Args:
            plan: The executed plan dictionary
            z_t: The original latent state vector
            u_t: The original world state dictionary (alias for original_world_state)
            original_world_state: The original world state dictionary (alias for u_t)
            score: The execution success score (alias for execution_success)
            execution_success: The execution success score (e.g. reduction in contradiction)
            contradiction_before: Contradiction score before execution (alias for contradictions_before)
            contradictions_before: Contradiction score before execution (alias for contradiction_before)
            contradiction_after: Contradiction score after execution (alias for contradictions_after)
            contradictions_after: Contradiction score after execution (alias for contradiction_after)
            **kwargs: Additional keyword arguments
        """
        # Handle parameter aliases
        world_state = original_world_state if original_world_state is not None else u_t
        success_score = execution_success if execution_success is not None else score
        before = contradictions_before if contradictions_before is not None else contradiction_before
        after = contradictions_after if contradictions_after is not None else contradiction_after
        
        # Extract plan source if available
        plan_source = plan.get('source', 'unknown') if plan else 'unknown'
        
        if plan_source not in self.planner_performance:
            self.planner_performance[plan_source] = {'success_rate': 0.5, 'weight': 1.0}
            
        # Calculate success metrics
        if before is not None and before > 0:
            reduction_percentage = (before - after) / before if after is not None else 0
        else:
            reduction_percentage = 0 if after and after > 0 else 1.0
            
        # Binary success: Did the plan reduce contradictions by at least 30%?
        successful = reduction_percentage >= 0.3
        
        # Update success rate using exponential moving average
        current_rate = self.planner_performance[plan_source]['success_rate']
        self.planner_performance[plan_source]['success_rate'] = (
            (1 - self.learning_rate) * current_rate + 
            self.learning_rate * (1.0 if successful else 0.0)
        )
        
        # Adjust weight based on success rate
        self.planner_performance[plan_source]['weight'] = (
            0.5 + self.planner_performance[plan_source]['success_rate']
        )
        
        # Log the original state and plan for future reference if available
        if z_t is not None or world_state is not None:
            state_info = {
                'latent_state_hash': hash(z_t.tobytes()) if z_t is not None else None,
                'world_state_hash': hash(str(world_state)) if world_state is not None else None
            }
            logger.debug(f"MetaReasoner learned from state: {state_info}")
        
        logger.info(f"MetaReasoner updated {plan_source} planner: " +
                   f"success_rate={self.planner_performance[plan_source]['success_rate']:.3f}, " +
                   f"weight={self.planner_performance[plan_source]['weight']:.3f}")
        
    def evaluate_plan(self, z_t: np.ndarray, plan: Dict, historical_context: List = None, 
                     current_world_state: Dict = None) -> float:
        """
        Evaluate a plan based on current state, historical context, and learned weights.
        
        Args:
            z_t: Current latent state vector
            plan: The plan to evaluate
            historical_context: Similar episodes from memory
            current_world_state: The current world state
            
        Returns:
            float: Improvement score for the plan (higher is better)
        """
        # Base evaluation
        base_score = 0.5
        
        # Apply learned weights if the plan has a source
        plan_source = plan.get('source')
        if plan_source in self.planner_performance:
            base_score *= self.planner_performance[plan_source]['weight']
            
        # Add contextual analysis
        if historical_context:
            # Add bonus for plans similar to historically successful ones
            similarity_bonus = 0.1
            base_score += similarity_bonus
            
        # Analyze contradiction risk
        if current_world_state:
            # For now, simple heuristic: higher solar power = lower risk
            solar_power = current_world_state.get('solar', 0)
            if solar_power > 70 and 'total_energy' in plan:
                # If plenty of solar and the plan isn't too energy intensive, boost score
                if plan['total_energy'] < solar_power * 0.8:
                    base_score += 0.2
                    
        return base_score
    
    def reframe_goal(self, u_t: Dict, z_t: np.ndarray, current_goal: Dict, contradiction_history: List[float]) -> Optional[Dict]:
        """
        Return a new goal if contradiction remains high over multiple cycles.
        Otherwise return None.

        Strategies:
        - Down-prioritize subgoals with unresolved tension
        - Propose abstract goal simplification
        - Shift from antagonistic to non-antagonistic framing

        Args:
            u_t: Current world state vector (unencoded)
            z_t: Current latent state vector (encoded)
            current_goal: The current goal structure
            contradiction_history: List of contradiction scores over recent cycles

        Returns:
            Optional[Dict]: A new goal structure if reframing is needed, None otherwise
        """
        # Check if contradiction remains persistently high
        if len(contradiction_history) < 3:
            return None
        
        avg_contradiction = sum(contradiction_history[-3:]) / 3
        
        # If average contradiction is low, no reframing needed
        if avg_contradiction <= 0.5:
            return None
            
        logger.info(f"MetaReasoner detected persistent contradiction: {avg_contradiction:.3f}")
        
        # Identify high-tension subgoals based on history
        high_tension_subgoals = []
        if len(self.history) >= 3:
            for entry in self.history[-3:]:
                if 'plan' in entry and 'steps' in entry['plan']:
                    for step in entry['plan']['steps']:
                        if step.get('tension', 0) > 0.5:
                            high_tension_subgoals.append(step.get('objective', ''))
        
        # Create a new goal by modifying the current one
        new_goal = copy.deepcopy(current_goal)
        
        # Strategy 1: Reduce precision of the current goal
        if avg_contradiction > 0.5:
            if 'target_state' in new_goal and isinstance(new_goal['target_state'], dict):
                for key in new_goal['target_state']:
                    # If it's a numeric value, make it more flexible with a range
                    if isinstance(new_goal['target_state'][key], (int, float)):
                        value = new_goal['target_state'][key]
                        # Convert precise values to flexible ranges
                        if 'constraints' not in new_goal:
                            new_goal['constraints'] = {}
                        if 'ranges' not in new_goal['constraints']:
                            new_goal['constraints']['ranges'] = {}
                        
                        # Add a 20% flexibility range
                        lower_bound = value * 0.8 if value > 0 else value * 1.2
                        upper_bound = value * 1.2 if value > 0 else value * 0.8
                        new_goal['constraints']['ranges'][key] = [lower_bound, upper_bound]
                        
                        # Mark the original value as a preference rather than a hard constraint
                        if 'preferences' not in new_goal:
                            new_goal['preferences'] = {}
                        new_goal['preferences'][key] = value
            
            # If goal has objectives field, soften language
            if 'objectives' in new_goal and isinstance(new_goal['objectives'], list):
                for i, objective in enumerate(new_goal['objectives']):
                    if isinstance(objective, str):
                        # Soften antagonistic language
                        new_goal['objectives'][i] = objective.replace("maximize", "increase")
                        new_goal['objectives'][i] = new_goal['objectives'][i].replace("minimize", "reduce")
                        
                        # Replace absolute terms with relative ones
                        new_goal['objectives'][i] = new_goal['objectives'][i].replace("ensure", "improve")
                        new_goal['objectives'][i] = new_goal['objectives'][i].replace("must", "should")
        
        # Strategy 2: Down-prioritize or remove subgoals with unresolved tension
        if high_tension_subgoals and 'subgoals' in new_goal:
            modified_subgoals = []
            for subgoal in new_goal['subgoals']:
                # Check if this subgoal is causing tension
                is_high_tension = any(tension_goal in subgoal.get('description', '') 
                                    for tension_goal in high_tension_subgoals)
                
                if is_high_tension:
                    # Option 1: Soften the subgoal by making it optional
                    softened_subgoal = copy.deepcopy(subgoal)
                    softened_subgoal['priority'] = 'low' if 'priority' in softened_subgoal else 'optional'
                    
                    # Add a note about why this was softened
                    if 'metadata' not in softened_subgoal:
                        softened_subgoal['metadata'] = {}
                    softened_subgoal['metadata']['reframed'] = True
                    softened_subgoal['metadata']['reason'] = "Persistent contradiction detected"
                    
                    modified_subgoals.append(softened_subgoal)
                else:
                    modified_subgoals.append(subgoal)
            
            # Update subgoals list
            new_goal['subgoals'] = modified_subgoals
            
        # Strategy 3: Shift from antagonistic to non-antagonistic framing
        if 'approach' in new_goal:
            antagonistic_terms = ["compete", "win", "beat", "outperform", "defeat", "conquer"]
            cooperative_terms = ["collaborate", "coordinate", "harmonize", "align", "cooperate", "share"]
            
            for i, term in enumerate(antagonistic_terms):
                if term in new_goal['approach']:
                    new_goal['approach'] = new_goal['approach'].replace(term, cooperative_terms[i])
        
        # Only return if changes were made
        if new_goal != current_goal:
            return new_goal
        
        return None

    def suggest_plan_improvements(self, plan: dict, z_t: np.ndarray) -> dict:
        """
        Suggests improvements to a candidate plan based on the current latent state vector.
        
        Args:
            plan: A candidate plan
            z_t: The current latent state vector
            
        Returns:
            dict: A dictionary containing a normalized score and a list of recommendations
        """
        # Initialize result structure
        result = {
            "score": 0.5,  # Default score
            "recommendations": []
        }
        
        # Apply heuristics to evaluate and improve the plan
        
        # Heuristic 1: Penalize overly high energy usage
        energy_usage = plan.get('total_energy', 0)
        if energy_usage > 0.8:
            result["score"] -= 0.2
            result["recommendations"].append(f"reduce total_energy (currently {energy_usage:.2f})")
        
        # Heuristic 2: Reward plans that reduce contradiction dimensions
        if hasattr(z_t, 'shape') and z_t.shape[0] > 0:
            # Assuming z_t[0] represents main contradiction dimension
            if plan.get('predicted_state_delta') is not None:
                delta_z = plan['predicted_state_delta']
                if delta_z[0] < 0:  # Reducing contradiction
                    result["score"] += 0.3
                    result["recommendations"].append(f"good reduction in contradiction dimension (Δz[0]={delta_z[0]:.2f})")
            else:
                # If no state delta prediction is available
                result["recommendations"].append("add state transition prediction to enable contradiction analysis")
        
        # Heuristic 3: Check resource allocation efficiency
        if 'steps' in plan:
            # Track which agents are used for which tasks
            agent_tasks = {}
            for step in plan['steps']:
                agent_id = step.get('agent_id')
                task_type = step.get('type')
                if agent_id and task_type:
                    if agent_id not in agent_tasks:
                        agent_tasks[agent_id] = []
                    agent_tasks[agent_id].append(task_type)
            
            # Check for irrigation assignments
            for agent_id, tasks in agent_tasks.items():
                if 'irrigation' in tasks and agent_id != 'AgriBot1' and 'AgriBot1' in agent_tasks:
                    result["score"] -= 0.1
                    result["recommendations"].append(f"prefer AgriBot1 for irrigation tasks (currently using {agent_id})")
        
        # Heuristic 4: Check for temporal optimization
        if 'steps' in plan:
            # Check if high-energy tasks are scheduled during peak solar availability
            high_energy_steps = [step for step in plan['steps'] if step.get('energy', 0) > 0.5]
            solar_peak_steps = [step for step in plan['steps'] if step.get('time_of_day', '') == 'noon']
            
            if high_energy_steps and not solar_peak_steps:
                result["score"] -= 0.15
                result["recommendations"].append("schedule high-energy tasks during solar peak times")
        
        # Normalize final score to be between 0 and 1
        result["score"] = max(0, min(1, result["score"]))
        
        # Log the suggestions
        logger.info(f"MetaReasoner plan improvement suggestions: score={result['score']:.2f}, " +
                   f"recommendations={result['recommendations']}")
        
        return result