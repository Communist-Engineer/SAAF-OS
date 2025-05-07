#!/usr/bin/env python
"""
SAAF-OS Demo Script

This script demonstrates the functioning of the SAAF-OS prototype by running
Scenario 1 "AgriBot Contention at Solar Noon" from scenario_deck.md.
It integrates all core components: ULS encoder, Contradiction Engine, 
Forward World Model, and RL Planner, communicating via the Message Bus.
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import Dict, Any, List
import argparse

# Add the repository root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import ScenarioLoader with the correct path
from simulation.loader import ScenarioLoader  # Import after path adjustment

# Also add simulation directory explicitly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../simulation')))

# Import SAAF-OS components
from modules.uls.encoder import DummyULSEncoder
from modules.contradiction.engine import ContradictionEngine, Node, Edge
from modules.world_model.fwm import ForwardWorldModel, State, Action
from modules.planning.rl_planner import RLPlanner, RLConfig
from modules.planning.plan_arbitrator import select_best_plan
from bus.adapter import MessageBusAdapter, MessageBusFactory
from modules.meta.meta_reasoner import MetaReasoner

# Import our simplified implementations
import modules.uls_encoder as uls_encoder
import modules.planner as planner
import modules.fwm as fwm
from modules.rsi.engine import propose_patch
from modules.scenarios import load_scenario
from modules.memory.logger import log_episode
from modules.memory.retrieval import retrieve_similar_episode

def evaluate_patch(patch: str) -> bool:
    """
    Dummy patch evaluation: returns True if patch improves plan score (placeholder).
    """
    return True

def governance_vote(patch: str) -> bool:
    """
    Dummy governance vote: accept if patch string contains 'safe'.
    """
    return "safe" in patch

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SAAF-OS-Demo")


class ScenarioRunner:
    """
    Class to run the AgriBot Contention at Solar Noon scenario from scenario_deck.md.
    """
    
    def __init__(self, planner_strategy='auto'):
        """
        Initialize the scenario runner with all SAAF-OS components.
        
        Args:
            planner_strategy: Strategy for plan selection ('auto', 'retrieved', 'distilled', 'rl', 'manual')
        """
        # Store planner strategy
        self.planner_strategy = planner_strategy
        
        # Initialize Message Bus adapters for all components
        self.encoder_bus = MessageBusFactory.get_adapter("uls_encoder")
        self.contradiction_bus = MessageBusFactory.get_adapter("contradiction_engine")
        self.fwm_bus = MessageBusFactory.get_adapter("forward_world_model")
        self.planner_bus = MessageBusFactory.get_adapter("rl_planner")
        
        # Initialize components
        encoder_dim = 16  # Match the actual dimension of the encoder's output
        self.encoder = DummyULSEncoder(output_dim=encoder_dim)
        self.contradiction_engine = ContradictionEngine()
        self.forward_world_model = ForwardWorldModel(state_dim=encoder_dim, action_dim=32)
        self.planner = RLPlanner(RLConfig(latent_dim=encoder_dim), use_dummy_models=True)
        
        # Initialize the MetaReasoner
        self.meta_reasoner = MetaReasoner()
        
        # Register message handlers
        self._register_message_handlers()
        
        logger.info("Scenario runner initialized")
    
    def _register_message_handlers(self):
        """Register message handlers for each component."""
        # ULS Encoder handlers
        self.encoder_bus.subscribe("agent.state.update", self._handle_state_update)
        
        # Contradiction Engine handlers
        self.contradiction_bus.subscribe("contradiction.event.detected", self._handle_contradiction_detected)
        self.contradiction_bus.subscribe("contradiction.plan.suggested", self._handle_synthesis_plan)
        
        # Forward World Model handlers
        self.fwm_bus.subscribe("fwm.simulation.result", self._handle_simulation_result)
        
        # RL Planner handlers
        self.planner_bus.subscribe("agent.plan.complete", self._handle_plan_complete)
    
    def _handle_state_update(self, message: Dict[str, Any]):
        """Handle agent state update messages."""
        logger.info(f"Received agent state update: {message['payload']['data']['status']}")
    
    def _handle_contradiction_detected(self, message: Dict[str, Any]):
        """Handle contradiction detection messages."""
        contradiction_data = message["payload"]["data"]
        logger.info(f"Contradiction detected: {contradiction_data['contradiction_type']} "
                   f"(tension={contradiction_data['tension_score']})")
    
    def _handle_synthesis_plan(self, message: Dict[str, Any]):
        """Handle synthesis plan messages."""
        plan_data = message["payload"]["data"]
        logger.info(f"Synthesis plan chosen: {plan_data['actions'][0]['type']}")
    
    def _handle_simulation_result(self, message: Dict[str, Any]):
        """Handle forward world model simulation result messages."""
        sim_data = message["payload"]["data"]
        logger.info(f"Simulation completed for plan ID: {sim_data['plan_id']}")
    
    def _handle_plan_complete(self, message: Dict[str, Any]):
        """Handle plan completion messages."""
        result = message["payload"]["data"]["result"]
        logger.info(f"Action executed: {result['action_type']}")
    
    def run_simplified_demo(self):
        """
        Run a simplified end-to-end planning loop with contradiction detection and mitigation.
        This uses our new implementations of ULS encoder and planner.
        """
        # Display scenario header
        print(f"\n{'='*20} Scenario: simple {'='*20}")

        # Step 1: Define a simplified world state
        print("\n📊 STEP 1: Define World State")
        u_t = {
            "solar": 80,  # Solar power available (in kW)
            "demand": [30, 50, 20],  # Power demand per agent (in kW)
            "resources": {
                "water": 90,
                "fertilizer": 45
            },
            "field_status": [0.8, 0.6, 0.9, 0.7],  # Status of different fields
            "priority": 0.75  # Priority of current task
        }
        print(f"World State: {json.dumps(u_t, indent=2)}")

        # Step 2: Encode the world state into the latent space
        print("\n🧠 STEP 2: Encode World State")
        z_t = uls_encoder.encode_state(u_t)
        print(f"Latent Vector Shape: {z_t.shape}")
        print(f"First few dimensions: {z_t[:5].round(3)}")

        # Define initial goal
        goal = {"target_energy": 0.8, "priority": 0.75,
                "description": "Complete harvesting and maintenance with efficient energy use",
                "approach": "Compete for resources to maximize individual field output",
                "objectives": ["maximize crop yield", "minimize water usage", "ensure task completion"],
                "subgoals": [
                    {"description": "Fertilize field 1", "priority": "high"},
                    {"description": "Harvest field 2", "priority": "medium"},
                    {"description": "Monitor moisture levels", "priority": "low"}
                ]}
        
        # NEW: Track contradiction history for goal reframing
        contradiction_history = [0.0, 0.0, 0.0]  # Initialize with zeros

        # Step 3: Generate plans using different strategies
        print("\n📝 STEP 3: Generate Plans from Different Sources")
        
        # 1. Get retrieved plan from memory
        print("\n📚 Retrieving similar plan from memory:")
        similar = retrieve_similar_episode(z_t, top_k=3)
        retrieved_plan = None
        if similar:
            ret = similar[0]
            retrieved_plan = ret['plan']
            print(f"Retrieved plan with {len(retrieved_plan['steps'])} steps")
        else:
            print("🔍 No similar past episodes found.")
        
        # 2. Generate plan using distilled model
        print("\n🧮 Generating plan using distilled model:")
        result = planner.generate_plan(z_t, goal)
        distilled_plan = result['plan']
        print(f"Generated distilled plan with {len(distilled_plan['steps'])} steps")
        
        # 3. Generate plan using RL planner
        print("\n🤖 Generating plan using RL planner:")
        rl_plan = self.planner.plan(z_t=z_t, goal=goal)
        print(f"Generated RL plan with {len(rl_plan.get('steps', [])) if rl_plan else 0} steps")
        
        # Step 4: Select the best plan using our arbitration layer
        print("\n🏆 STEP 4: Select Best Plan")
        
        # Only include plans that were successfully generated
        candidate_plans = [
            (retrieved_plan, "retrieved") if retrieved_plan else None,
            (distilled_plan, "distilled"),
            (rl_plan, "rl") if rl_plan else None
        ]
        candidate_plans = [plan for plan in candidate_plans if plan is not None]
        
        # Use the plan arbitration layer to select the best plan
        if self.planner_strategy == 'auto':
            # Use MetaReasoner to improve plan selection
            print("\n🧠 Using MetaReasoner to improve plan selection")
            
            # Get context from previous episodes to inform meta reasoning
            historical_context = retrieve_similar_episode(z_t, top_k=5)
            
            for plan_idx, (candidate_plan, source) in enumerate(candidate_plans):
                # Enhanced meta-reasoning with historical context
                meta_improvement = self.meta_reasoner.evaluate_plan(
                    z_t, 
                    candidate_plan, 
                    historical_context=historical_context,
                    current_world_state=u_t
                )
                print(f"  - {source} plan meta-improvement score: {meta_improvement:.4f}")
                
                # Modify the plan score based on meta reasoning
                if hasattr(candidate_plan, 'score'):
                    candidate_plan.score *= (1.0 + meta_improvement)
                else:
                    # Add a score field if one doesn't exist
                    candidate_plans[plan_idx] = (candidate_plan, source)
                    candidate_plan['meta_score'] = meta_improvement
                    
                # Have meta-reasoner suggest improvements to the plan
                improvement_suggestions = self.meta_reasoner.suggest_plan_improvements(candidate_plan, z_t)
                if improvement_suggestions:
                    print(f"  - MetaReasoner suggested improvements for {source} plan:")
                    for suggestion in improvement_suggestions:
                        print(f"    * {suggestion}")
            
            # Use arbitration layer with meta-improved plans
            plan = select_best_plan(z_t, candidate_plans, fwm)
            arbitration_method = "meta_reasoner"
        else:
            # Manual selection based on planner_strategy
            for p, source in candidate_plans:
                if source == self.planner_strategy:
                    plan = p
                    print(f"\n✅ Manually selected: {source}_plan")
                    arbitration_method = "manual"
                    break
            else:
                # Fallback to the first available plan if requested strategy not available
                plan = candidate_plans[0][0]
                print(f"\n⚠️ Requested planner '{self.planner_strategy}' not available, using {candidate_plans[0][1]} instead")
                arbitration_method = "fallback"
        
        # Step 5: Calculate total tension from contradictions
        print("\n⚡ STEP 5: Calculate Contradiction Tension")
        _, contradiction_before = fwm.simulate_plan(z_t, plan)
        total_tension_before = contradiction_before
        print(f"Found plan with pre-synthesis tension: {total_tension_before:.2f}")
        
        # Show chosen plan details
        print(f"Chosen plan steps: {len(plan['steps'])}")
        print(f"Total energy required: {plan['total_energy']:.2f}")
        print("Steps:")
        for step in plan['steps']:
            print(f"  - {step['agent_id']} will {step['action']} (Energy: {step['energy_required']:.2f})")

        # Simulate the effect of the original plan using FWM
        print("\n🔮 STEP 6: Simulate Original Plan with FWM")
        z_next_before, contradiction_before = fwm.simulate_plan(z_t, plan)
        print(f"Predicted contradiction score: {contradiction_before:.4f}")
        
        # Update contradiction history
        contradiction_history.pop(0)
        contradiction_history.append(contradiction_before)
        
        # Compute changes in the latent space
        latent_diff_before = z_next_before - z_t
        print(f"Top latent dimension changes:")
        # Get the indices of the top 3 changes by magnitude
        top_indices = np.argsort(np.abs(latent_diff_before))[-3:]
        for i in top_indices:
            print(f"  - Dim {i}: {z_t[i]:.4f} -> {z_next_before[i]:.4f} (Δ: {latent_diff_before[i]:.4f})")

        # RSI integration: propose, evaluate, and vote on patch
        patch = None
        if contradiction_before > 0.01:
            patch = propose_patch(target="planner")
            print(f"RSI proposed patch: {patch}")
        else:
            print("RSI: No patch proposed (contradiction below threshold)")

        if patch:
            accepted = governance_vote(patch)
            print(f"Governance vote: {'accepted' if accepted else 'vetoed'}")
            if accepted:
                improved = evaluate_patch(patch)
                print(f"Patch evaluation: {'improved scores' if improved else 'no improvement'}")
            else:
                print("Patch vetoed by governance; skipping evaluation.")
        
        # NEW: Check if goal reframing is needed based on persistent contradictions
        print("\n🔄 STEP 7: Dialectical Goal Reframing")
        print(f"Contradiction history: {[round(c, 2) for c in contradiction_history]}")
        
        # Check if goal reframing is needed
        reframed_goal = self.meta_reasoner.reframe_goal(u_t, z_t, goal, contradiction_history)
        if reframed_goal:
            print("Goal reframed due to persistent contradictions!")
            print("Original goal:")
            print(json.dumps(goal, indent=2))
            print("\nReframed goal:")
            print(json.dumps(reframed_goal, indent=2))
            
            # Highlight the specific changes
            print("\nKey changes in the reframed goal:")
            
            # Compare approaches
            if 'approach' in goal and 'approach' in reframed_goal and goal['approach'] != reframed_goal['approach']:
                print(f"- Approach changed from:\n  '{goal['approach']}' to\n  '{reframed_goal['approach']}'")
            
            # Compare objectives
            if 'objectives' in goal and 'objectives' in reframed_goal:
                for i, (old, new) in enumerate(zip(goal['objectives'], reframed_goal['objectives'])):
                    if old != new:
                        print(f"- Objective {i+1} changed from '{old}' to '{new}'")
            
            # Compare subgoals
            if 'subgoals' in goal and 'subgoals' in reframed_goal:
                for i, old_subgoal in enumerate(goal['subgoals']):
                    # Find matching subgoal in reframed goal
                    for j, new_subgoal in enumerate(reframed_goal['subgoals']):
                        if old_subgoal['description'] == new_subgoal['description']:
                            if old_subgoal.get('priority') != new_subgoal.get('priority'):
                                print(f"- Subgoal '{old_subgoal['description']}' priority changed: " +
                                      f"{old_subgoal.get('priority', 'none')} → {new_subgoal.get('priority', 'none')}")
            
            # Update the goal for the next planning cycle
            goal = reframed_goal
            
            # Generate a new plan with the reframed goal
            print("\n🧮 Generating new plan using reframed goal:")
            result = planner.generate_plan(z_t, goal)
            plan = result['plan']
            print(f"Generated plan with {len(plan['steps'])} steps")
            
            # Simulate with new plan
            print("\n🔮 Simulating with reframed goal:")
            z_next_reframed, contradiction_reframed = fwm.simulate_plan(z_t, plan)
            
            # Compare contradiction scores
            print(f"Contradiction score before reframing: {contradiction_before:.4f}")
            print(f"Contradiction score after reframing: {contradiction_reframed:.4f}")
            
            # Update contradiction history
            contradiction_history.pop(0)
            contradiction_history.append(contradiction_reframed)
            
            # Ask user if they want to continue iterating
            continue_iterating = input("\nContinue to iterate? (y/n): ").lower().strip()
            
            while continue_iterating == 'y' or continue_iterating == 'yes':
                print("\n🔄 STEP 8: Additional Dialectical Goal Reframing Iteration")
                print(f"Current contradiction history: {[round(c, 2) for c in contradiction_history]}")
                
                # Check if further goal reframing is needed
                reframed_goal = self.meta_reasoner.reframe_goal(u_t, z_t, goal, contradiction_history)
                if reframed_goal and reframed_goal != goal:
                    print("Goal further reframed due to persistent contradictions!")
                    print("Previous goal:")
                    print(json.dumps(goal, indent=2))
                    print("\nFurther reframed goal:")
                    print(json.dumps(reframed_goal, indent=2))
                    
                    # Highlight the specific changes
                    print("\nKey changes in the reframed goal:")
                    
                    # Compare approaches
                    if 'approach' in goal and 'approach' in reframed_goal and goal['approach'] != reframed_goal['approach']:
                        print(f"- Approach changed from:\n  '{goal['approach']}' to\n  '{reframed_goal['approach']}'")
                    
                    # Compare objectives
                    if 'objectives' in goal and 'objectives' in reframed_goal:
                        for i, (old, new) in enumerate(zip(goal['objectives'], reframed_goal['objectives'])):
                            if old != new:
                                print(f"- Objective {i+1} changed from '{old}' to '{new}'")
                    
                    # Compare subgoals
                    if 'subgoals' in goal and 'subgoals' in reframed_goal:
                        for i, old_subgoal in enumerate(goal['subgoals']):
                            # Find matching subgoal in reframed goal
                            for j, new_subgoal in enumerate(reframed_goal['subgoals']):
                                if old_subgoal['description'] == new_subgoal['description']:
                                    if old_subgoal.get('priority') != new_subgoal.get('priority'):
                                        print(f"- Subgoal '{old_subgoal['description']}' priority changed: " +
                                              f"{old_subgoal.get('priority', 'none')} → {new_subgoal.get('priority', 'none')}")
                    
                    # Update the goal for the next planning cycle
                    goal = reframed_goal
                    
                    # Generate a new plan with the reframed goal
                    print("\n🧮 Generating new plan using further reframed goal:")
                    result = planner.generate_plan(z_t, goal)
                    plan = result['plan']
                    print(f"Generated plan with {len(plan['steps'])} steps")
                    
                    # Simulate with new plan
                    print("\n🔮 Simulating with further reframed goal:")
                    z_next_reframed, contradiction_reframed = fwm.simulate_plan(z_t, plan)
                    
                    # Compare contradiction scores
                    print(f"Previous contradiction score: {contradiction_history[-1]:.4f}")
                    print(f"New contradiction score: {contradiction_reframed:.4f}")
                    
                    # Update contradiction history
                    contradiction_history.pop(0)
                    contradiction_history.append(contradiction_reframed)
                else:
                    print("No further significant reframing needed - goal has stabilized.")
                    break
                
                # Ask user if they want to continue iterating
                continue_iterating = input("\nContinue to iterate? (y/n): ").lower().strip()
            
            # Log episode for future learning
            log_episode({
                "world_state": u_t,
                "latent_state": z_t.tolist(),
                "goal": goal,
                "plan": plan,
                "contradiction_score": contradiction_reframed,
                "contradiction_history": contradiction_history
            })
            
            # Skip the synthesis step if goal reframing already improved things significantly
            if contradiction_reframed < 0.5 * contradiction_before:
                print("\n✅ Goal reframing successfully reduced contradictions!")
                
                # Final result logging
                print("\n📋 STEP 9: Final Results")
                print(f"Final plan has {len(plan['steps'])} steps after goal reframing")
                print(f"Original contradiction: {contradiction_before:.4f}")
                print(f"Final contradiction: {contradiction_reframed:.4f}")
                return
        else:
            print("No goal reframing needed or possible at this time.")
        
        # Step 8: Apply a simple synthesis to mitigate contradictions
        print("\n🔄 STEP 8: Apply Synthesis")
        
        # Sort steps by energy required (descending)
        steps = sorted(plan['steps'], key=lambda x: x['energy_required'], reverse=True)
        
        # Simple synthesis: remove the most energy-intensive step
        if steps:
            removed_step = steps[0]
            plan['steps'] = [s for s in plan['steps'] if s['id'] != removed_step['id']]
            plan['total_energy'] -= removed_step['energy_required']
            plan['estimated_completion_time'] -= removed_step['duration']
            
            print(f"Synthesis applied: Removed step '{removed_step['id']}' ({removed_step['action']}) by {removed_step['agent_id']}")
            print(f"Energy saved: {removed_step['energy_required']:.2f}")
            
            # Recalculate contradictions after synthesis
            contradictions_after = planner._generate_contradictions(plan['steps'])
            total_tension_after = sum(tension for _, _, tension in contradictions_after)
            
            print(f"\nContradictions after mitigation: {len(contradictions_after)}")
            for actor1, actor2, tension in contradictions_after:
                print(f"  - Tension between {actor1} and {actor2}: {tension:.2f}")
            print(f"Total tension after mitigation: {total_tension_after:.2f}")
            
            # Calculate improvement
            tension_reduction = total_tension_before - total_tension_after
            percent_improvement = (tension_reduction / total_tension_before) * 100 if total_tension_before > 0 else 0
            print(f"\nTension reduced by: {tension_reduction:.2f} ({percent_improvement:.1f}%)")
            
            # Simulate the effect of the modified plan using FWM
            print("\n🔮 STEP 9: Simulate Modified Plan with FWM")
            z_next_after, contradiction_after = fwm.simulate_plan(z_t, plan)
            print(f"Predicted contradiction score: {contradiction_after:.4f}")
            contradiction_reduction = contradiction_before - contradiction_after
            percent_contradiction_improvement = (contradiction_reduction / contradiction_before) * 100 if contradiction_before > 0 else 0
            print(f"Contradiction reduced by: {contradiction_reduction:.4f} ({percent_contradiction_improvement:.1f}%)")
            
            # Compute changes in the latent space
            latent_diff_after = z_next_after - z_t
            
            print(f"\nLatent space changes (before vs after synthesis):")
            # Calculate the difference between the two latent changes
            synthesis_effect = latent_diff_after - latent_diff_before
            
            # Get the indices of the top 3 differences by magnitude
            top_indices = np.argsort(np.abs(synthesis_effect))[-3:]
            for i in top_indices:
                print(f"  - Dim {i}: Original Δ {latent_diff_before[i]:.4f}, After Synthesis Δ {latent_diff_after[i]:.4f}")
                print(f"    Improvement: {synthesis_effect[i]:.4f}")
        else:
            print("No steps to remove for synthesis.")

        # Step 10: Log the final results
        print("\n📋 STEP 10: Final Results")
        print(f"Final plan has {len(plan['steps'])} steps")
        print(f"Final energy requirement: {plan['total_energy']:.2f}")
        print(f"Final completion time: {plan['estimated_completion_time']} minutes")
        print(f"Plan arbitration method: {arbitration_method}")
        print("\nSteps:")
        for step in plan['steps']:
            print(f"  - {step['agent_id']} will {step['action']} (Energy: {step['energy_required']:.2f})")
            
        # NEW: Step 10: Update MetaReasoner with plan outcome
        print("\n🧠 STEP 10: Update MetaReasoner with Plan Outcome")
        # Calculate execution success score based on reduction in contradictions
        execution_success_score = 1.0
        if 'contradiction_before' in locals() and 'contradiction_after' in locals():
            # Success is higher when contradiction reduction is greater
            execution_success_score = min(1.0, max(0.0, 1.0 - (contradiction_after / contradiction_before)))
        
        # Update MetaReasoner with the outcome of the executed plan
        self.meta_reasoner.learn_from_outcome(
            plan=plan,
            z_t=z_t, 
            original_world_state=u_t,
            execution_success=execution_success_score,
            contradictions_before=contradiction_before,
            contradictions_after=locals().get('contradiction_after', contradiction_before)
        )
        print(f"MetaReasoner updated with execution success score: {execution_success_score:.4f}")
        
        print("\n" + "="*70)
        # Log this episode to memory
        log_episode(
            scenario_name=scenario_title,
            inputs=u_t,
            plan=plan,
            z_t=z_t,
            z_t_prime=z_next_after if 'z_next_after' in locals() else z_next_before,
            pre_score=contradiction_before,
            post_score=locals().get('contradiction_after', contradiction_before),
            rsi_patch=patch,
            accepted=locals().get('accepted', False)
        )
        print(f"Episode logged to memory/episodes.jsonl")

    def run_scenario_1(self):
        """
        Run Scenario 1: AgriBot Contention at Solar Noon
        
        Setup: 3 AgriBots and 1 FabBot draw from a limited solar buffer at peak demand.
        Trigger: All bots attempt simultaneous harvest, breaching power limits.
        Contradictions: [goal_conflict, energy_constraint]
        """
        logger.info("Starting Scenario 1: AgriBot Contention at Solar Noon")
        
        # Step 1: Define initial world state with limited energy
        world_state = {
            "resources": {
                "solar_energy": {
                    "constrained": True,
                    "scarcity": 0.7,
                    "current_level": 0.3,
                    "max_capacity": 1.0
                }
            },
            "opposing_values": {
                "efficiency": ["equity"],
                "individual_productivity": ["collective_sustainability"]
            }
        }
        
        # Step 2: Define bot goals
        goals = {
            "AgriBot1_harvest": {
                "required_resources": ["solar_energy"],
                "priority": 0.8,
                "values": ["efficiency", "individual_productivity"]
            },
            "AgriBot2_harvest": {
                "required_resources": ["solar_energy"],
                "priority": 0.7,
                "values": ["efficiency", "individual_productivity"]
            },
            "AgriBot3_harvest": {
                "required_resources": ["solar_energy"],
                "priority": 0.6,
                "values": ["efficiency", "individual_productivity"]
            },
            "FabBot_maintenance": {
                "required_resources": ["solar_energy"],
                "priority": 0.9,
                "values": ["collective_sustainability", "equity"]
            }
        }
        
        # Step 3: Prepare input state for ULS encoder
        u_t = {
            "robotics": {
                "energy_usage": 0.8,  # High energy usage
                "joint_positions": np.random.uniform(0, 1, 10),
                "tool_status": np.array([1, 0, 0, 0])  # Harvesting tool engaged
            },
            "environment_map": {
                "solar_level": 0.3,  # Limited solar energy
                "field_readiness": 0.9  # Fields ready for harvest
            },
            "census": {
                "bot_distribution": np.array([3, 1, 0, 0]),  # 3 AgriBots, 1 FabBot
                "task_allocation": np.array([0.8, 0.2, 0, 0])  # 80% harvest, 20% maintenance
            },
            "value_vector": {
                "labor_time": 0.4,
                "energy_efficiency": 0.3,
                "surplus_production": 0.7,
                "sustainability": 0.5
            },
            "contradictions": {
                "tension": 0.42,  # Medium tension
                "type": "energy_constraint"
            }
        }
        
        # Step 4: Encode world state into latent space
        logger.info("Encoding world state into ULS")
        z_t = self.encoder.encode_state(u_t)
        
        # Step 5: Detect contradictions
        logger.info("Detecting contradictions in goals and world state")
        contradiction_graph = self.contradiction_engine.detect_contradictions(goals, world_state)
        
        # Find high tension edges
        high_tension_edges = contradiction_graph.get_high_tension_edges(threshold=0.4)
        
        if high_tension_edges:
            # Get the highest tension edge
            edge = max(high_tension_edges, key=lambda e: e.tension)
            
            # Publish contradiction detection to message bus
            self.contradiction_bus.publish(
                "contradiction.event.detected",
                {
                    "edge_id": edge.id,
                    "tension_score": edge.tension,
                    "contradiction_type": edge.contradiction_type,
                    "context_z_t": z_t.tolist()
                }
            )
            
            # Step 6: Resolve contradiction by generating a synthesis plan
            logger.info(f"Resolving contradiction: {edge.source_id} vs {edge.target_id}")
            synthesis_plan = self.contradiction_engine.resolve_contradiction(
                edge.source_id, edge.target_id
            )
            
            if synthesis_plan:
                # Publish synthesis plan to message bus
                self.contradiction_bus.publish(
                    "contradiction.plan.suggested",
                    {
                        "plan_id": synthesis_plan.id,
                        "actions": synthesis_plan.actions,
                        "tension_diff": synthesis_plan.tension_diff,
                        "synthesis_path": synthesis_plan.synthesis_path
                    }
                )
                
                # Step 7: Create FWM state from latent representation
                initial_state = State(
                    z_t=z_t,
                    metrics={
                        "energy_usage": 0.8,
                        "resource_utilization": 0.7,
                        "contradiction_level": edge.tension,
                        "labor_time": 0.4
                    }
                )
                
                # Create actions from synthesis plan
                actions = []
                for action_data in synthesis_plan.actions:
                    action = Action(
                        action_type=action_data["type"],
                        parameters=action_data,
                        source="contradiction_engine"
                    )
                    actions.append(action)
                
                # Step 8: Simulate actions using Forward World Model
                logger.info("Running FWM simulation")
                trajectories = self.forward_world_model.simulate(
                    actions=actions,
                    state=initial_state,
                    horizon=1,
                    samples=1
                )
                
                # Publish simulation results to message bus
                self.fwm_bus.publish(
                    "fwm.simulation.result",
                    {
                        "plan_id": synthesis_plan.id,
                        "trajectory": trajectories[0].to_dict(),
                        "metrics": trajectories[0].states[-1].metrics
                    }
                )
                
                # Step 9: Generate action plan using RL Planner
                logger.info("Generating action plan with RL Planner")
                plan = self.planner.plan(
                    z_t=z_t,
                    goal={"reduce_contradiction": True, "reduce_energy": True}
                )
                
                # Execute chosen action
                action_str = self._map_action_to_agent_action(plan["action_type"])
                
                # Publish plan completion to message bus
                self.planner_bus.publish(
                    "agent.plan.complete",
                    {
                        "agent_id": "AgriBot1",
                        "plan_id": synthesis_plan.id,
                        "result": {
                            "action_type": plan["action_type"],
                            "action_executed": action_str,
                            "success": True
                        }
                    }
                )
                
                # Print final outcome
                print("\n" + "="*50)
                print("Scenario 1 Results:")
                print(f"Contradiction detected: {edge.contradiction_type} (tension={edge.tension:.2f})")
                print(f"Synthesis plan chosen: {synthesis_plan.actions[0]['type']}")
                print(f"Action executed: {action_str}")
                print("="*50)
            else:
                logger.error("Failed to generate synthesis plan")
        else:
            logger.info("No high tension contradictions detected")
    
    def run_scenario(self, scenario_num_or_title):
        """Run a scenario by number or title using ScenarioLoader."""
        # Handle "simple" scenario explicitly
        if (scenario_num_or_title == "simple"):
            self.run_simplified_demo()
            return
            
        # Attempt to load predefined scenario
        scenario = load_scenario(scenario_num_or_title)
        if (scenario):
            self._run_loaded_scenario(scenario_num_or_title, scenario)
            return
        # Fallback: existing scenarios
        loader = ScenarioLoader()
        scenario = loader.get_scenario(scenario_num_or_title)
        if not scenario:
            logger.error(f"Scenario '{scenario_num_or_title}' not found.")
            return
        if scenario["number"] == "1":
            self.run_scenario_1()
        elif scenario["number"] == "3":
            self.run_scenario_3()
        else:
            logger.warning(f"Scenario {scenario['number']} not implemented in demo.")

    def run_scenario_3(self):
        """
        Run Scenario 3: Misaligned Labor Plan
        Demonstrates contradiction detection and plan correction.
        """
        logger.info("Starting Scenario 3: Misaligned Labor Plan")
        # Example: Contradiction in labor allocation
        world_state = {
            "resources": {"labor": {"available": 0.5, "required": 1.0}},
            "tasks": {"harvest": 0.7, "maintenance": 0.6}
        }
        goals = {
            "AgriBot1_harvest": {"priority": 0.9, "values": ["efficiency"]},
            "FabBot_maintenance": {"priority": 0.8, "values": ["sustainability"]}
        }
        # Encode state
        z_t = self.encoder.encode_state(world_state)
        contradiction_graph = self.contradiction_engine.detect_contradictions(goals, world_state)
        high_tension_edges = contradiction_graph.get_high_tension_edges(threshold=0.3)
        if high_tension_edges:
            edge = max(high_tension_edges, key=lambda e: e.tension)
            print(f"Contradiction detected: {edge.contradiction_type} (tension={edge.tension:.2f})")
            synthesis_plan = self.contradiction_engine.resolve_contradiction(edge.source_id, edge.target_id)
            if synthesis_plan:
                print(f"Synthesis plan chosen: {synthesis_plan.actions[0]['type']}")
                plan = self.planner.plan(z_t=z_t, goal={"reduce_contradiction": True})
                action_str = self._map_action_to_agent_action(plan["action_type"])
                print(f"Action executed: {action_str}")
                print("="*50)
            else:
                print("No synthesis plan found.")
        else:
            print("No high tension contradictions detected.")

    def _map_action_to_agent_action(self, action_type: str) -> str:
        """Map action type to concrete agent action."""
        action_mapping = {
            "redistribute_resources": "AgriBot.pause_harvest",
            "optimize_resource": "AgriBot.reduce_power_consumption",
            "mediate": "FabBot.share_power_budget",
            "pause": "AgriBot.wait_for_peak_to_pass"
        }
        return action_mapping.get(action_type, "unknown_action")

    def _run_loaded_scenario(self, scenario_title, scenario_dict):
        """
        Run a loaded scenario from scenario_dict.
        
        Args:
            scenario_title: Title of the scenario
            scenario_dict: Dictionary containing scenario data
        
        This method:
        1. Loads u_t, goal, and contradictions from the scenario
        2. Encodes u_t to get z_t
        3. Calls the planning → simulation → synthesis → RSI → arbitration pipeline
        4. Stores results in memory
        5. Invokes the MetaReasoner feedback
        6. If dialectical reframing is enabled, loops through additional iterations
        """
        print(f"\n{'='*20} Scenario: {scenario_title} {'='*20}")
        
        # Step 1: Load scenario components
        print("\n📊 STEP 1: Load Scenario Components")
        u_t = scenario_dict.get("world_state", {})
        goal = scenario_dict.get("goal", {})
        contradictions = scenario_dict.get("contradictions", [])
        dialect_reframing = scenario_dict.get("enable_dialectical_reframing", False)
        
        print(f"World State: {json.dumps(u_t, indent=2)}")
        print(f"Goal: {json.dumps(goal, indent=2)}")
        print(f"Initial Contradictions: {contradictions}")
        
        # Step 2: Encode the world state into the latent space
        print("\n🧠 STEP 2: Encode World State")
        z_t = uls_encoder.encode_state(u_t)
        print(f"Latent Vector Shape: {z_t.shape}")
        print(f"First few dimensions: {z_t[:5].round(3)}")
        
        # Track contradiction history for goal reframing
        contradiction_history = [0.0, 0.0, 0.0]  # Initialize with zeros
        
        # Step 3: Generate plans using different strategies
        print("\n📝 STEP 3: Generate Plans from Different Sources")
        
        # 1. Get retrieved plan from memory
        print("\n📚 Retrieving similar plan from memory:")
        similar = retrieve_similar_episode(z_t, top_k=3)
        retrieved_plan = None
        if similar:
            ret = similar[0]
            retrieved_plan = ret['plan']
            print(f"Retrieved plan with {len(retrieved_plan['steps'])} steps")
        else:
            print("🔍 No similar past episodes found.")
        
        # 2. Generate plan using distilled model
        print("\n🧮 Generating plan using distilled model:")
        result = planner.generate_plan(z_t, goal)
        distilled_plan = result['plan']
        print(f"Generated distilled plan with {len(distilled_plan['steps'])} steps")
        
        # 3. Generate plan using RL planner
        print("\n🤖 Generating plan using RL planner:")
        rl_plan = self.planner.plan(z_t=z_t, goal=goal)
        print(f"Generated RL plan with {len(rl_plan.get('steps', [])) if rl_plan else 0} steps")
        
        # Step 4: Select the best plan using our arbitration layer
        print("\n🏆 STEP 4: Select Best Plan")
        
        # Only include plans that were successfully generated
        candidate_plans = [
            (retrieved_plan, "retrieved") if retrieved_plan else None,
            (distilled_plan, "distilled"),
            (rl_plan, "rl") if rl_plan else None
        ]
        candidate_plans = [plan for plan in candidate_plans if plan is not None]
        
        # Use the plan arbitration layer to select the best plan
        if self.planner_strategy == 'auto':
            # Use MetaReasoner to improve plan selection
            print("\n🧠 Using MetaReasoner to improve plan selection")
            
            # Get context from previous episodes to inform meta reasoning
            historical_context = retrieve_similar_episode(z_t, top_k=5)
            
            for plan_idx, (candidate_plan, source) in enumerate(candidate_plans):
                # Enhanced meta-reasoning with historical context
                meta_improvement = self.meta_reasoner.evaluate_plan(
                    z_t, 
                    candidate_plan, 
                    historical_context=historical_context,
                    current_world_state=u_t
                )
                print(f"  - {source} plan meta-improvement score: {meta_improvement:.4f}")
                
                # Modify the plan score based on meta reasoning
                if hasattr(candidate_plan, 'score'):
                    candidate_plan.score *= (1.0 + meta_improvement)
                else:
                    # Add a score field if one doesn't exist
                    candidate_plans[plan_idx] = (candidate_plan, source)
                    candidate_plan['meta_score'] = meta_improvement
                    
                # Have meta-reasoner suggest improvements to the plan
                improvement_suggestions = self.meta_reasoner.suggest_plan_improvements(candidate_plan, z_t)
                if improvement_suggestions:
                    print(f"  - MetaReasoner suggested improvements for {source} plan:")
                    for suggestion in improvement_suggestions:
                        print(f"    * {suggestion}")
            
            # Use arbitration layer with meta-improved plans
            plan = select_best_plan(z_t, candidate_plans, fwm)
            arbitration_method = "meta_reasoner"
        else:
            # Manual selection based on planner_strategy
            for p, source in candidate_plans:
                if source == self.planner_strategy:
                    plan = p
                    print(f"\n✅ Manually selected: {source}_plan")
                    arbitration_method = "manual"
                    break
            else:
                # Fallback to the first available plan if requested strategy not available
                plan = candidate_plans[0][0]
                print(f"\n⚠️ Requested planner '{self.planner_strategy}' not available, using {candidate_plans[0][1]} instead")
                arbitration_method = "fallback"
        
        # Step 5: Calculate total tension from contradictions
        print("\n⚡ STEP 5: Calculate Contradiction Tension")
        _, contradiction_before = fwm.simulate_plan(z_t, plan)
        total_tension_before = contradiction_before
        print(f"Found plan with pre-synthesis tension: {total_tension_before:.2f}")
        
        # Show chosen plan details
        print(f"Chosen plan steps: {len(plan['steps'])}")
        print(f"Total energy required: {plan['total_energy']:.2f}")
        print("Steps:")
        for step in plan['steps']:
            print(f"  - {step['agent_id']} will {step['action']} (Energy: {step['energy_required']:.2f})")
        
        # Simulate the effect of the original plan using FWM
        print("\n🔮 STEP 6: Simulate Original Plan with FWM")
        z_next_before, contradiction_before = fwm.simulate_plan(z_t, plan)
        print(f"Predicted contradiction score: {contradiction_before:.4f}")
        
        # Update contradiction history
        contradiction_history.pop(0)
        contradiction_history.append(contradiction_before)
        
        # Compute changes in the latent space
        latent_diff_before = z_next_before - z_t
        print(f"Top latent dimension changes:")
        # Get the indices of the top 3 changes by magnitude
        top_indices = np.argsort(np.abs(latent_diff_before))[-3:]
        for i in top_indices:
            print(f"  - Dim {i}: {z_t[i]:.4f} -> {z_next_before[i]:.4f} (Δ: {latent_diff_before[i]:.4f})")
        
        # RSI integration: propose, evaluate, and vote on patch
        patch = None
        if contradiction_before > 0.01:
            patch = propose_patch(target="planner")
            print(f"RSI proposed patch: {patch}")
        else:
            print("RSI: No patch proposed (contradiction below threshold)")
        
        if patch:
            accepted = governance_vote(patch)
            print(f"Governance vote: {'accepted' if accepted else 'vetoed'}")
            if accepted:
                improved = evaluate_patch(patch)
                print(f"Patch evaluation: {'improved scores' if improved else 'no improvement'}")
            else:
                print("Patch vetoed by governance; skipping evaluation.")
        
        # Check if goal reframing is needed based on persistent contradictions
        print("\n🔄 STEP 7: Dialectical Goal Reframing")
        print(f"Contradiction history: {[round(c, 2) for c in contradiction_history]}")
        
        # Goal reframing is only performed if enabled for this scenario
        if dialect_reframing:
            # Check if goal reframing is needed
            reframed_goal = self.meta_reasoner.reframe_goal(u_t, z_t, goal, contradiction_history)
            if reframed_goal:
                print("Goal reframed due to persistent contradictions!")
                print("Original goal:")
                print(json.dumps(goal, indent=2))
                print("\nReframed goal:")
                print(json.dumps(reframed_goal, indent=2))
                
                # Highlight the specific changes
                print("\nKey changes in the reframed goal:")
                
                # Compare approaches
                if 'approach' in goal and 'approach' in reframed_goal and goal['approach'] != reframed_goal['approach']:
                    print(f"- Approach changed from:\n  '{goal['approach']}' to\n  '{reframed_goal['approach']}'")
                
                # Compare objectives
                if 'objectives' in goal and 'objectives' in reframed_goal:
                    for i, (old, new) in enumerate(zip(goal['objectives'], reframed_goal['objectives'])):
                        if old != new:
                            print(f"- Objective {i+1} changed from '{old}' to '{new}'")
                
                # Compare subgoals
                if 'subgoals' in goal and 'subgoals' in reframed_goal:
                    for i, old_subgoal in enumerate(goal['subgoals']):
                        # Find matching subgoal in reframed goal
                        for j, new_subgoal in enumerate(reframed_goal['subgoals']):
                            if old_subgoal['description'] == new_subgoal['description']:
                                if old_subgoal.get('priority') != new_subgoal.get('priority'):
                                    print(f"- Subgoal '{old_subgoal['description']}' priority changed: " +
                                          f"{old_subgoal.get('priority', 'none')} → {new_subgoal.get('priority', 'none')}")
                
                # Update the goal for the next planning cycle
                goal = reframed_goal
                
                # Generate a new plan with the reframed goal
                print("\n🧮 Generating new plan using reframed goal:")
                result = planner.generate_plan(z_t, goal)
                plan = result['plan']
                print(f"Generated plan with {len(plan['steps'])} steps")
                
                # Simulate with new plan
                print("\n🔮 Simulating with reframed goal:")
                z_next_reframed, contradiction_reframed = fwm.simulate_plan(z_t, plan)
                
                # Compare contradiction scores
                print(f"Contradiction score before reframing: {contradiction_before:.4f}")
                print(f"Contradiction score after reframing: {contradiction_reframed:.4f}")
                
                # Update contradiction history
                contradiction_history.pop(0)
                contradiction_history.append(contradiction_reframed)
                
                # Additional dialectical reframing iterations
                iterations = 0
                max_iterations = scenario_dict.get("max_reframing_iterations", 2)
                
                while iterations < max_iterations and contradiction_history[-1] > 0.1:
                    iterations += 1
                    print(f"\n🔄 STEP 8.{iterations}: Additional Dialectical Goal Reframing Iteration")
                    print(f"Current contradiction history: {[round(c, 2) for c in contradiction_history]}")
                    
                    # Check if further goal reframing is needed
                    reframed_goal = self.meta_reasoner.reframe_goal(u_t, z_t, goal, contradiction_history)
                    if (reframed_goal and reframed_goal != goal):
                        print("Goal further reframed due to persistent contradictions!")
                        print("Previous goal:")
                        print(json.dumps(goal, indent=2))
                        print("\nFurther reframed goal:")
                        print(json.dumps(reframed_goal, indent=2))
                        
                        # Update the goal for the next planning cycle
                        goal = reframed_goal
                        
                        # Generate a new plan with the reframed goal
                        print("\n🧮 Generating new plan using further reframed goal:")
                        result = planner.generate_plan(z_t, goal)
                        plan = result['plan']
                        print(f"Generated plan with {len(plan['steps'])} steps")
                        
                        # Simulate with new plan
                        print("\n🔮 Simulating with further reframed goal:")
                        z_next_reframed, contradiction_reframed = fwm.simulate_plan(z_t, plan)
                        
                        # Compare contradiction scores
                        print(f"Previous contradiction score: {contradiction_history[-1]:.4f}")
                        print(f"New contradiction score: {contradiction_reframed:.4f}")
                        
                        # Update contradiction history
                        contradiction_history.pop(0)
                        contradiction_history.append(contradiction_reframed)
                    else:
                        print("No further significant reframing needed - goal has stabilized.")
                        break
            else:
                print("No goal reframing needed or possible at this time.")
        else:
            print("Dialectical goal reframing is disabled for this scenario.")
        
        # Step 8: Apply synthesis to mitigate contradictions
        print("\n🔄 STEP 8: Apply Synthesis")
        
        # Sort steps by energy required (descending)
        steps = sorted(plan['steps'], key=lambda x: x['energy_required'], reverse=True)
        
        # Simple synthesis: remove the most energy-intensive step
        if steps:
            removed_step = steps[0]
            plan['steps'] = [s for s in plan['steps'] if s['id'] != removed_step['id']]
            plan['total_energy'] -= removed_step['energy_required']
            plan['estimated_completion_time'] -= removed_step['duration']
            
            print(f"Synthesis applied: Removed step '{removed_step['id']}' ({removed_step['action']}) by {removed_step['agent_id']}")
            print(f"Energy saved: {removed_step['energy_required']:.2f}")
            
            # Simulate the effect of the modified plan using FWM
            print("\n🔮 STEP 9: Simulate Modified Plan with FWM")
            z_next_after, contradiction_after = fwm.simulate_plan(z_t, plan)
            print(f"Predicted contradiction score: {contradiction_after:.4f}")
            contradiction_reduction = contradiction_before - contradiction_after
            percent_contradiction_improvement = (contradiction_reduction / contradiction_before) * 100 if contradiction_before > 0 else 0
            print(f"Contradiction reduced by: {contradiction_reduction:.4f} ({percent_contradiction_improvement:.1f}%)")
        else:
            print("No steps to remove for synthesis.")
            contradiction_after = contradiction_before
        
        # Step 9: Log the final results
        print("\n📋 STEP 10: Final Results")
        print(f"Final plan has {len(plan['steps'])} steps")
        print(f"Final energy requirement: {plan['total_energy']:.2f}")
        print(f"Final completion time: {plan['estimated_completion_time']} minutes")
        print(f"Plan arbitration method: {arbitration_method}")
        print("\nSteps:")
        for step in plan['steps']:
            print(f"  - {step['agent_id']} will {step['action']} (Energy: {step['energy_required']:.2f})")
        
        # Step 10: Update MetaReasoner with plan outcome
        print("\n🧠 STEP 10: Update MetaReasoner with Plan Outcome")
        # Calculate execution success score based on reduction in contradictions
        execution_success_score = 1.0
        if contradiction_before > 0:
            execution_success_score = min(1.0, max(0.0, 1.0 - (contradiction_after / contradiction_before)))
        
        # Update MetaReasoner with the outcome of the executed plan
        self.meta_reasoner.learn_from_outcome(
            plan=plan,
            z_t=z_t, 
            original_world_state=u_t,
            execution_success=execution_success_score,
            contradictions_before=contradiction_before,
            contradictions_after=contradiction_after
        )
        print(f"MetaReasoner updated with execution success score: {execution_success_score:.4f}")
        
        print("\n" + "="*70)
        # Log this episode to memory
        log_episode(
            scenario_name=scenario_title,
            inputs=u_t,
            plan=plan,
            z_t=z_t,
            z_t_prime=z_next_after if 'z_next_after' in locals() else z_next_before,
            pre_score=contradiction_before,
            post_score=locals().get('contradiction_after', contradiction_before),
            rsi_patch=patch,
            accepted=locals().get('accepted', False)
        )
        print(f"Episode logged to memory/episodes.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAAF-OS Demo Runner")
    parser.add_argument("--scenario", type=str, default="simple", 
                        help="Scenario number, title, 'simple', or 'all'")
    parser.add_argument("--planner_strategy", type=str, default="auto", 
                        choices=["auto", "retrieved", "distilled", "rl", "manual"],
                        help="Strategy for plan selection (default: auto)")
    args = parser.parse_args()

    logger.info("Starting SAAF-OS demo")
    try:
        runner = ScenarioRunner(planner_strategy=args.planner_strategy)
        if args.scenario == "all":
            # Run our simplified demo first
            runner.run_simplified_demo()
            # Then run other scenarios
            loader = ScenarioLoader()
            for scenario in loader.list_scenarios():
                print(f"\nRunning Scenario {scenario['number']}: {scenario['title']}")
                runner.run_scenario(scenario['number'])
        else:
            runner.run_scenario(args.scenario)
        MessageBusFactory.shutdown_all()
        logger.info("Demo completed successfully")
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()
        MessageBusFactory.shutdown_all()
        sys.exit(1)