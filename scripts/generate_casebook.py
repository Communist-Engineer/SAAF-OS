import json
import os
from collections import defaultdict, Counter
import numpy as np

def parse_synthesis_log(path):
    cases = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            ctype = entry.get('contradiction_1', 'unknown')
            score = entry.get('resolution_score', 0.0)
            tension = abs(score) if isinstance(score, (int, float)) else 0.0
            cases[ctype].append({
                'score': score,
                'tension': tension,
                'strategy': entry.get('strategy'),
                'synthesis_path': entry.get('synthesis_path'),
            })
    return cases

def summarize_casebook(cases):
    summary = {}
    for ctype, entries in cases.items():
        scores = [e['score'] for e in entries]
        tensions = [e['tension'] for e in entries]
        strategies = Counter(e['strategy'] for e in entries)
        summary[ctype] = {
            'count': len(entries),
            'avg_tension': float(np.mean(tensions)) if tensions else None,
            'avg_resolution_score': float(np.mean(scores)) if scores else None,
            'strategies': dict(strategies),
            # Optionally: top 3 latent dims (if available)
        }
    return summary

def main():
    log_path = 'diagnostics/synthesis_log.jsonl'
    out_path = 'reports/casebook_summary.json'
    if not os.path.exists(log_path):
        print(f'No synthesis log at {log_path}')
        return
    cases = parse_synthesis_log(log_path)
    summary = summarize_casebook(cases)
    os.makedirs('reports', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Casebook summary written to {out_path}')

if __name__ == '__main__':
    main()
