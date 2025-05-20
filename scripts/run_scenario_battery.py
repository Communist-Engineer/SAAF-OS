import os
import glob
import zipfile
import subprocess
from datetime import datetime

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
    print(f'All scenario results zipped to {zip_path}')

def main():
    run_all_scenarios()

if __name__ == '__main__':
    main()
