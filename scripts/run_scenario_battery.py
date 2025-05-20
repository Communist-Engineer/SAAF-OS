import os
import glob
import zipfile
import subprocess
import json
from datetime import datetime

def evaluate_synthesis(scenario_name, diagnostics_path='diagnostics/synthesis_log.jsonl'):
    # Analyze synthesis log for this scenario
    contradictions = 0
    resolved = 0
    scores = []
    if not os.path.exists(diagnostics_path):
        return None
    with open(diagnostics_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if scenario_name in str(entry.get('synthesis_path', '')):
                contradictions += 1
                score = entry.get('resolution_score')
                if score and score > 0.5:
                    resolved += 1
                if score is not None:
                    scores.append(score)
    percent_resolved = (resolved / contradictions) * 100 if contradictions else 0
    avg_score = float(sum(scores) / len(scores)) if scores else None
    summary = {
        'contradictions_detected': contradictions,
        'percent_resolved': percent_resolved,
        'avg_resolution_score': avg_score
    }
    outpath = os.path.join('reports', f'synthesis_eval_{scenario_name}.json')
    os.makedirs('reports', exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Synthesis evaluation for {scenario_name} written to {outpath}')
    return summary

def run_all_scenarios():
    scenario_names = ['simple', 'scenario_1', 'scenario_2']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', f'saaf_battery_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    summary_paths = []
    for scenario in scenario_names:
        print(f'Running scenario: {scenario}')
        result = subprocess.run(['python', 'scripts/run_demo.py', '--scenario', scenario, '--output', os.path.join(results_dir, f'{scenario}_summary.json')], capture_output=True, text=True)
        print(result.stdout)
        summary_path = os.path.join(results_dir, f'{scenario}_summary.json')
        if os.path.exists(summary_path):
            summary_paths.append(summary_path)
        else:
            print(f'Warning: No summary found for {scenario}')
        evaluate_synthesis(scenario)
    # Copy episodes.jsonl and audit log if present
    if os.path.exists('memory/episodes.jsonl'):
        import shutil
        shutil.copy('memory/episodes.jsonl', os.path.join(results_dir, 'episodes.jsonl'))
    if os.path.exists('governance/audit_log.jsonl'):
        import shutil
        shutil.copy('governance/audit_log.jsonl', os.path.join(results_dir, 'audit_log.jsonl'))
    # Zip all results
    zip_path = results_dir + '.zip'
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(results_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, results_dir)
                zipf.write(file_path, arcname)
        # Also add reports/*.json and plots/*.png
        for f in glob.glob('reports/*.json'):
            zipf.write(f, os.path.join('reports', os.path.basename(f)))
        for f in glob.glob('plots/*.png'):
            zipf.write(f, os.path.join('plots', os.path.basename(f)))
    print(f'All scenario results zipped to {zip_path}')

def main():
    run_all_scenarios()

if __name__ == '__main__':
    main()
