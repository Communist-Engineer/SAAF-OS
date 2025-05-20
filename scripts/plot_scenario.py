import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_episodes(path):
    episodes = []
    with open(path, 'r') as f:
        for line in f:
            try:
                episodes.append(json.loads(line))
            except Exception:
                continue
    return episodes

def plot_contradiction(episodes, outdir):
    contradiction = [e.get('contradiction_score') for e in episodes if 'contradiction_score' in e]
    plt.figure()
    plt.plot(contradiction, marker='o')
    plt.title('Contradiction Level Over Time')
    plt.xlabel('Step')
    plt.ylabel('Contradiction Score')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'contradiction_level.png'))
    plt.savefig(os.path.join(outdir, 'contradiction_level.svg'))
    plt.close()

def plot_reward(episodes, outdir):
    rewards = [e.get('reward') for e in episodes if 'reward' in e]
    if rewards:
        plt.figure()
        plt.plot(rewards, marker='x', color='green')
        plt.title('Predicted Reward Over Time')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'predicted_reward.png'))
        plt.savefig(os.path.join(outdir, 'predicted_reward.svg'))
        plt.close()

def plot_delta_z(episodes, outdir):
    z_t = [e.get('z_t') for e in episodes if 'z_t' in e]
    z_t_prime = [e.get('z_t_prime') for e in episodes if 'z_t_prime' in e]
    if z_t and z_t_prime and len(z_t) == len(z_t_prime):
        delta = [np.linalg.norm(np.array(z2) - np.array(z1)) for z1, z2 in zip(z_t, z_t_prime)]
        plt.figure()
        plt.plot(delta, marker='s', color='purple')
        plt.title('Δz_t Magnitude Over Time')
        plt.xlabel('Step')
        plt.ylabel('||z_t+1 - z_t||')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'delta_z.png'))
        plt.savefig(os.path.join(outdir, 'delta_z.svg'))
        plt.close()

def plot_latent_heatmap(episodes, outdir):
    # Optional: plot heatmap if latent_summary or z_t available
    z_t = [e.get('z_t') for e in episodes if 'z_t' in e]
    if z_t:
        arr = np.array(z_t)
        plt.figure(figsize=(10, 4))
        plt.imshow(arr.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Latent Value')
        plt.title('Latent State Evolution (z_t)')
        plt.xlabel('Step')
        plt.ylabel('Latent Dimension')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'latent_heatmap.png'))
        plt.savefig(os.path.join(outdir, 'latent_heatmap.svg'))
        plt.close()

def main():
    if len(sys.argv) < 2:
        print('Usage: python plot_scenario.py <episodes.jsonl> [output_dir]')
        sys.exit(1)
    episodes_path = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else 'reports/plots_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(outdir, exist_ok=True)
    episodes = load_episodes(episodes_path)
    plot_contradiction(episodes, outdir)
    plot_reward(episodes, outdir)
    plot_delta_z(episodes, outdir)
    plot_latent_heatmap(episodes, outdir)
    print(f'Plots saved to {outdir}')

if __name__ == '__main__':
    main()
