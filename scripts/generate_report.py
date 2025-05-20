import os
import sys
import json
import argparse
import numpy as np
from collections import Counter, defaultdict

def parse_episodes(episodes_path, scenario_name=None):
    contradiction_losses = []
    rewards = []
    patch_attempts = 0
    patch_approvals = 0
    governance_outcomes = []
    value_vectors = []
    value_vector_before = None
    value_vector_after = None
    patch_rejections = 0
    patch_rejection_reasons = []
    with open(episodes_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Filter by scenario if needed
            if scenario_name and entry.get('scenario_name') != scenario_name:
                continue
            # Contradiction loss
            if 'contradiction_score' in entry:
                contradiction_losses.append(entry['contradiction_score'])
            if 'reward' in entry:
                rewards.append(entry['reward'])
            # Patch attempts/approvals
            if 'rsi_patch' in entry:
                patch_attempts += 1
                if entry.get('accepted'):
                    patch_approvals += 1
                else:
                    patch_rejections += 1
                    if 'rejection_reason' in entry:
                        patch_rejection_reasons.append(entry['rejection_reason'])
            # Governance outcomes
            if 'governance_vote' in entry:
                governance_outcomes.append(entry['governance_vote'])
            # Value vectors
            if 'latent_state' in entry:
                value_vectors.append(entry['latent_state'])
                if value_vector_before is None:
                    value_vector_before = entry['latent_state']
                value_vector_after = entry['latent_state']
    # Compute stats
    avg_contradiction = float(np.mean(contradiction_losses)) if contradiction_losses else None
    mean_reward = float(np.mean(rewards)) if rewards else None
    std_reward = float(np.std(rewards)) if rewards else None
    approval_rate = patch_approvals / patch_attempts if patch_attempts else None
    governance_summary = dict(Counter(governance_outcomes))
    value_drift = None
    if value_vector_before is not None and value_vector_after is not None:
        value_drift = float(np.linalg.norm(np.array(value_vector_after) - np.array(value_vector_before)))
    return {
        'avg_contradiction_loss': avg_contradiction,
        'patch_attempt_count': patch_attempts,
        'patch_approval_rate': approval_rate,
        'patch_rejection_count': patch_rejections,
        'patch_rejection_reasons': patch_rejection_reasons,
        'governance_outcome_summary': governance_summary,
        'final_value_drift': value_drift,
        'mean_reward': mean_reward,
        'reward_std': std_reward
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default='memory/episodes.jsonl')
    parser.add_argument('--scenario', default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    scenario_name = args.scenario or 'default'
    report = parse_episodes(args.episodes, scenario_name)
    output_path = args.output or f'reports/{scenario_name}_summary.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Report written to {output_path}')

if __name__ == '__main__':
    main()
