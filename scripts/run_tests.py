"""
SAAF-OS Testing Harness
Loads all scenarios, runs tests, outputs JSON summary per testing_harness.md.
"""
import os
import sys
import json
import subprocess
from simulation.loader import ScenarioLoader

def run_pytest(test_path):
    """Run pytest on a given path and return (success, output)."""
    result = subprocess.run([
        sys.executable, '-m', 'pytest', test_path, '--json-report', '--json-report-file=pytest_report.json'
    ], capture_output=True, text=True)
    try:
        with open('pytest_report.json') as f:
            report = json.load(f)
    except Exception:
        report = None
    return result.returncode == 0, report

def run_scenarios():
    loader = ScenarioLoader()
    results = []
    for scenario in loader.list_scenarios():
        scenario_id = scenario['number']
        scenario_title = scenario['title']
        print(f"Running scenario {scenario_id}: {scenario_title}")
        proc = subprocess.run([
            sys.executable, 'scripts/run_demo.py', '--scenario', scenario_id
        ], capture_output=True, text=True)
        results.append({
            'scenario': scenario_id,
            'title': scenario_title,
            'returncode': proc.returncode,
            'stdout': proc.stdout,
            'stderr': proc.stderr
        })
    return results

def main():
    summary = {}
    # Run all unit tests
    print("Running unit tests...")
    unit_success, unit_report = run_pytest('tests')
    summary['unit_tests'] = {
        'success': unit_success,
        'report': unit_report
    }
    # Run all scenarios
    print("Running scenario tests...")
    scenario_results = run_scenarios()
    summary['scenarios'] = scenario_results
    # Write summary
    with open('test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nTest summary written to test_summary.json")
    # Print pass/fail
    if not unit_success or any(r['returncode'] != 0 for r in scenario_results):
        print("\nSome tests failed.")
        sys.exit(1)
    print("\nAll tests passed.")

if __name__ == "__main__":
    main()
