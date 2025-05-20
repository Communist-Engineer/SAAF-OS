import os
import glob
import zipfile
import subprocess

def run_all_scenarios():
    # Assume scenario runner is run_demo.py and scenarios are hardcoded or listed
    scenario_names = ['simple', 'scenario_1', 'scenario_2']
    summary_paths = []
    for scenario in scenario_names:
        print(f'Running scenario: {scenario}')
        # Run the scenario using run_demo.py with scenario argument if supported
        result = subprocess.run(['python', 'scripts/run_demo.py', '--scenario', scenario], capture_output=True, text=True)
        print(result.stdout)
        summary_path = os.path.join('reports', f'{scenario}_summary.json')
        if os.path.exists(summary_path):
            summary_paths.append(summary_path)
        else:
            print(f'Warning: No summary found for {scenario}')
    # Zip all summaries
    zip_path = os.path.join('reports', 'scenario_battery_results.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for path in summary_paths:
            zipf.write(path, os.path.basename(path))
    print(f'All scenario summaries zipped to {zip_path}')

def main():
    run_all_scenarios()

if __name__ == '__main__':
    main()
