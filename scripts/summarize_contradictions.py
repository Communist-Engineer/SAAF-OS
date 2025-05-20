import json
import os
from collections import Counter, defaultdict
import numpy as np

def main():
    path = 'diagnostics/contradiction_broadcasts.jsonl'
    out_path = 'reports/contradiction_summary.json'
    if not os.path.exists(path):
        print(f'No contradiction broadcasts at {path}')
        return
    by_type = Counter()
    by_agent = Counter()
    times = defaultdict(list)
    synth_times = []
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                ctype = entry.get('c_type', 'unknown')
                agent = entry.get('agent_id', 'unknown')
                ts = entry.get('timestamp')
                by_type[ctype] += 1
                by_agent[agent] += 1
                times[ctype].append(ts)
            except Exception:
                continue
    # Avg time between contradiction and synthesis (dummy: just time between events)
    avg_time = {k: float(np.mean(np.diff(sorted(v)))) if len(v) > 1 else None for k, v in times.items()}
    summary = {
        'contradiction_counts': dict(by_type),
        'top_agents': dict(by_agent.most_common(5)),
        'avg_time_between_events': avg_time
    }
    os.makedirs('reports', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Contradiction summary written to {out_path}')

if __name__ == '__main__':
    main()
