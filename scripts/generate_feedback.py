import json
import os
from collections import defaultdict

def load_episodes(path):
    episodes = []
    with open(path, 'r') as f:
        for line in f:
            try:
                episodes.append(json.loads(line))
            except Exception:
                continue
    return episodes

def load_synthesis_log(path):
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, 'r') as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries

def generate_feedback(scenario, episodes, synthesis_log):
    insights = []
    # Example: Find first unresolved contradiction
    for i, e in enumerate(episodes):
        if e.get('contradiction_score', 0) > 0.5 and not e.get('rsi_accepted', True):
            insights.append(f"Planner failed to resolve contradiction until step {i}")
            break
    # Example: Patch family drift
    for entry in synthesis_log:
        if 'patch_family' in str(entry.get('synthesis_path', '')) and entry.get('resolution_score', 0) < 0.5:
            insights.append(f"RSI patch family {entry.get('synthesis_path')} showed alignment drift")
    # Example: Governance veto
    for i, e in enumerate(episodes):
        if e.get('rsi_accepted') is False:
            insights.append(f"Governance veto at step {i} delayed resolution")
    return {
        "scenario": scenario,
        "insights": insights or ["No major issues detected."]
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default='memory/episodes.jsonl')
    parser.add_argument('--synthesis', default='diagnostics/synthesis_log.jsonl')
    parser.add_argument('--scenario', default='scenario')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    episodes = load_episodes(args.episodes)
    synthesis_log = load_synthesis_log(args.synthesis)
    feedback = generate_feedback(args.scenario, episodes, synthesis_log)
    outpath = args.output or f'reports/feedback_{args.scenario}.json'
    os.makedirs('reports', exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(feedback, f, indent=2)
    print(f'Feedback written to {outpath}')

if __name__ == '__main__':
    main()
