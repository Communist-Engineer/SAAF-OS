import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

def load_audit_log(path):
    entries = []
    if not os.path.exists(path):
        print(f"No audit log at {path}")
        return entries
    with open(path, 'r') as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries

def summarize_audit(entries, scenario=None):
    proposals = 0
    approved = 0
    vetoed = 0
    value_deltas = []
    times = []
    for e in entries:
        if scenario and scenario not in str(e.get('proposal', '')):
            continue
        proposals += 1
        if e.get('decision') == 'approved':
            approved += 1
        elif e.get('decision') == 'rejected':
            vetoed += 1
        v1 = e.get('value_vector_before')
        v2 = e.get('value_vector_after')
        if v1 is not None and v2 is not None:
            try:
                import numpy as np
                delta = float(np.linalg.norm(np.array(v2) - np.array(v1)))
                value_deltas.append(delta)
            except Exception:
                pass
        times.append(e.get('timestamp'))
    print(f"Proposals: {proposals}")
    print(f"Approved: {approved}")
    print(f"Vetoed: {vetoed}")
    if value_deltas:
        print(f"Mean value vector delta: {sum(value_deltas)/len(value_deltas):.4f}")
    if times:
        times = sorted([t for t in times if t is not None])
        if len(times) > 1:
            duration = times[-1] - times[0]
            print(f"Audit span: {datetime.fromtimestamp(times[0])} to {datetime.fromtimestamp(times[-1])} ({duration:.1f} sec)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--scenario', default=None)
    args = parser.parse_args()
    entries = load_audit_log('governance/audit_log.jsonl')
    if args.summary or args.scenario:
        summarize_audit(entries, scenario=args.scenario)
    else:
        for e in entries:
            print(json.dumps(e, indent=2))

if __name__ == '__main__':
    main()
