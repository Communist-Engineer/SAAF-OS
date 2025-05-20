import json
import os
from collections import Counter, defaultdict

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

def load_audit_log(path):
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

def print_markdown_summary():
    synth = load_synthesis_log('diagnostics/synthesis_log.jsonl')
    audit = load_audit_log('governance/audit_log.jsonl')
    by_type = defaultdict(list)
    for e in synth:
        by_type[e.get('contradiction_1','unknown')].append(e)
    print('| Contradiction Type | Count | Avg. Tension | Resolution Rate | Top Strategies |')
    print('|--------------------|-------|--------------|-----------------|---------------|')
    for ctype, entries in by_type.items():
        count = len(entries)
        avg_tension = sum(abs(e.get('resolution_score',0)) for e in entries)/count if count else 0
        resolved = sum(1 for e in entries if e.get('resolution_score',0)>0.5)
        rate = f'{(resolved/count*100):.1f}%' if count else '0.0%'
        strategies = Counter(e.get('strategy') for e in entries)
        top_strat = ', '.join([f'{k}({v})' for k,v in strategies.most_common(2)])
        print(f'| {ctype} | {count} | {avg_tension:.2f} | {rate} | {top_strat} |')
    # Governance vetoes
    vetoes = sum(1 for e in audit if e.get('decision') == 'rejected')
    print(f'\n**Total governance vetoes:** {vetoes}')

if __name__ == '__main__':
    print_markdown_summary()
