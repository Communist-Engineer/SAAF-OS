#!/usr/bin/env python
"""
Stress test all predefined scenarios to compute average tensions, energy savings, and RSI success rates.
"""
import json
import numpy as np
from modules.scenarios import load_scenario
import modules.uls_encoder as uls_encoder
import modules.planner as planner
import modules.fwm as fwm
from modules.rsi.engine import propose_patch
from modules.memory.retrieval import retrieve_similar_episode

def governance_vote(patch: str) -> bool:
    return 'safe' in patch

def stress_scenarios(runs_per_scenario: int = 3):
    scenario_names = ['solar_conflict', 'veto_loop', 'alienation_drift']
    summary = {}

    for name in scenario_names:
        tensions = []
        energy_savings = []
        rsi_successes = 0
        total_rsi = 0
        memory_reuse = 0
        reuse_deltas = []  # new_score - retrieved_score
        total_runs = 0
        for _ in range(runs_per_scenario):
            scenario = load_scenario(name)
            if scenario is None:
                continue
            u_t = scenario['u_t']
            z_t = uls_encoder.encode_state(u_t)
            result = planner.generate_plan(z_t, scenario['goal'])
            plan = result['plan']
            
            # Get contradiction scores from the result dictionary
            contradiction_before = result.get('contradiction_before', 0)
            contradiction_after = result.get('contradiction_after', 0)
            
            # initial tension from scenario definitions
            pre_tension = sum(t for _, _, t in scenario['contradictions'])
            # memory retrieval and plan comparison
            similar = retrieve_similar_episode(z_t, top_k=1)
            retrieved_score = float('inf')
            if similar:
                retrieved_plan = similar[0]['plan']
                _, retrieved_score = fwm.simulate_plan(z_t, retrieved_plan)
            # simulate new plan
            z_next, new_score = fwm.simulate_plan(z_t, plan)
            total_runs += 1
            # memory reuse decision
            if retrieved_score < new_score:
                memory_reuse += 1
                reuse_deltas.append(new_score - retrieved_score)
            # synthesis: remove highest-energy step
            steps = sorted(plan['steps'], key=lambda s: s['energy_required'], reverse=True)
            if steps:
                removed = steps[0]
                energy_before = plan['total_energy']
                plan['steps'] = [s for s in plan['steps'] if s['id'] != removed['id']]
                plan['total_energy'] -= removed['energy_required']
                energy_after = plan['total_energy']
            else:
                energy_before = energy_after = 0
            # contradictions after
            contradictions_after = planner._generate_contradictions(plan['steps'])
            post_tension = sum(t for _, _, t in contradictions_after)
            tensions.append(pre_tension - post_tension)
            energy_savings.append(energy_before - energy_after)
            # RSI
            if contradiction_before > 0.01:  # Using contradiction_before instead of pre_score
                total_rsi += 1
                patch = propose_patch(target='planner')
                if governance_vote(patch):
                    rsi_successes += 1
        # compute averages
        avg_tension_red = np.mean(tensions) if tensions else 0
        avg_energy_save = np.mean(energy_savings) if energy_savings else 0
        rsi_rate = (rsi_successes / total_rsi * 100) if total_rsi else 0
        reuse_rate = (memory_reuse / total_runs * 100) if total_runs else 0
        avg_reuse_delta = np.mean(reuse_deltas) if reuse_deltas else 0
        summary[name] = {
            'avg_tension_reduction': float(avg_tension_red),
            'avg_energy_saving': float(avg_energy_save),
            'rsi_success_rate': float(rsi_rate),
            'memory_reuse_rate': float(reuse_rate),
            'avg_memory_delta': float(avg_reuse_delta)
        }

    # Print summary table
    print("\nStress Test Summary:")
    print(f"{'Scenario':<20} {'ΔTension':>10} {'ΔEnergy':>10} {'RSI %':>8} {'Reuse %':>8} {'Mem Δ':>8}")
    for name, m in summary.items():
        print(
            f"{name:<20} {m['avg_tension_reduction']:>10.3f} {m['avg_energy_saving']:>10.3f} "
            f"{m['rsi_success_rate']:>8.1f} {m['memory_reuse_rate']:>8.1f} {m['avg_memory_delta']:>8.3f}"
        )
    
    # Print overall averages
    print("\nOverall Averages:")
    avg_tension = np.mean([m['avg_tension_reduction'] for m in summary.values()])
    avg_energy = np.mean([m['avg_energy_saving'] for m in summary.values()])
    avg_rsi = np.mean([m['rsi_success_rate'] for m in summary.values()])
    print(f"Average Tension Reduction: {avg_tension:.3f}")
    print(f"Average Energy Savings: {avg_energy:.3f}")
    print(f"Average RSI Patch Success Rate: {avg_rsi:.2f}%")

if __name__ == '__main__':
    stress_scenarios()