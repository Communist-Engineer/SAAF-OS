#!/usr/bin/env python
"""
Memory Explorer for SAAF-OS episodes
"""
import argparse
import json
import os
from typing import List, Dict, Any, Tuple

def load_episodes(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    episodes = []
    with open(path, 'r') as f:
        for line in f:
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return episodes

def apply_filters(episodes: List[Dict[str, Any]], filters: List[str]) -> List[Dict[str, Any]]:
    result = episodes
    for flt in filters:
        if '=' not in flt:
            continue
        key, val = flt.split('=', 1)
        key = key.strip()
        val = val.strip()
        if key == 'scenario':
            result = [e for e in result if e.get('scenario') == val]
        elif key == 'rsi_accepted':
            b = val.lower() in ('1','true','yes')
            result = [e for e in result if e.get('rsi_accepted') == b]
        elif key == 'tension_min':
            m = float(val)
            result = [e for e in result if (e['pre_score'] - e['post_score']) >= m]
        elif key == 'tension_max':
            M = float(val)
            result = [e for e in result if (e['pre_score'] - e['post_score']) <= M]
    return result

def sort_episodes(episodes: List[Dict[str, Any]], sort_key: str) -> List[Dict[str, Any]]:
    if sort_key == 'tension_reduction':
        return sorted(episodes, key=lambda e: e['pre_score'] - e['post_score'], reverse=True)
    if sort_key == 'energy_saving':
        return sorted(episodes, key=lambda e: e.get('energy_initial',0) - e.get('energy_final',0), reverse=True)
    return episodes

def summarize_episode(e: Dict[str, Any]) -> Tuple[str, float, float, bool]:
    scen = e.get('scenario','')
    tension_red = e['pre_score'] - e['post_score']
    energy_save = e.get('energy_initial',0) - e.get('energy_final',0)
    rsi = e.get('rsi_accepted', False)
    return scen, tension_red, energy_save, rsi

def main():
    parser = argparse.ArgumentParser(description='View logged memory episodes')
    parser.add_argument('--filter', action='append', default=[],
                        help='Filter expressions: scenario=name, rsi_accepted=true, tension_min=0.1, tension_max=0.5')
    parser.add_argument('--sort', choices=['tension_reduction','energy_saving'],
                        help='Sort key')
    parser.add_argument('--top', type=int, default=None, help='Show top N episodes')
    parser.add_argument('--path', type=str, default='memory/episodes.jsonl', help='Path to episodes JSONL')
    args = parser.parse_args()

    episodes = load_episodes(args.path)
    if not episodes:
        print('No episodes found at', args.path)
        return

    # Apply filters
    episodes = apply_filters(episodes, args.filter)
    # Sort
    if args.sort:
        episodes = sort_episodes(episodes, args.sort)
    # Top
    if args.top:
        episodes = episodes[:args.top]

    # Display summary table
    print(f"{'Scenario':<20} {'ΔTension':>10} {'ΔEnergy':>10} {'RSI?':>6}")
    for e in episodes:
        scen, tr, es, rsi = summarize_episode(e)
        print(f"{scen:<20} {tr:>10.3f} {es:>10.3f} {str(rsi):>6}")

if __name__ == '__main__':
    main()