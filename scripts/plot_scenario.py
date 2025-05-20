import os
import json
import sys
import matplotlib.pyplot as plt

def plot_scenario_metrics(summary_path):
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    steps = list(range(len(summary.get('contradiction_scores', []))))
    contradiction_scores = summary.get('contradiction_scores', [])
    value_vectors = summary.get('value_vectors', [])
    rewards = summary.get('rewards', [])
    # Plot contradiction scores
    plt.figure(figsize=(10, 6))
    plt.plot(steps, contradiction_scores, marker='o', label='Contradiction Score')
    plt.xlabel('Step')
    plt.ylabel('Contradiction Score')
    plt.title('Contradiction Score Over Scenario')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Optionally plot value vector drift if available
    if value_vectors:
        plt.figure(figsize=(10, 6))
        for i in range(len(value_vectors[0])):
            plt.plot(steps, [v[i] for v in value_vectors], label=f'Value Dim {i}')
        plt.xlabel('Step')
        plt.ylabel('Value Vector Component')
        plt.title('Value Vector Drift Over Scenario')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    # Optionally plot rewards if available
    if rewards:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, rewards, marker='x', label='Reward')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Reward Over Scenario')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    if len(sys.argv) < 2:
        print('Usage: python plot_scenario.py <summary_json_path>')
        sys.exit(1)
    plot_scenario_metrics(sys.argv[1])

if __name__ == '__main__':
    main()
