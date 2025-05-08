#!/bin/bash

timestamp=$(date +%Y%m%d_%H%M%S)
snapshot_dir="saaf_os_diag_$timestamp"
mkdir -p "$snapshot_dir"/{tests,demo_logs,models,memory,diagnostics}

echo "🧪 Running full test suite..."
pytest -v --tb=short > "$snapshot_dir/diagnostics/pytest.log"

echo "🧬 Running scenario simulations..."
SCENARIOS=("solar_conflict" "veto_loop" "alienation_drift")
for scenario in "${SCENARIOS[@]}"; do
    echo "▶️ Running scenario: $scenario"
    python3 -m scripts.run_demo --scenario "$scenario" \
        > "$snapshot_dir/demo_logs/$scenario.log" 2>&1
done

echo "🧠 Copying memory logs..."
cp memory/*.jsonl "$snapshot_dir/memory/" 2>/dev/null
cp memory/*.log "$snapshot_dir/memory/" 2>/dev/null

echo "🧠 Copying trained models..."
cp models/*.pt "$snapshot_dir/models/" 2>/dev/null

echo "📊 Saving environment metadata..."
git rev-parse HEAD > "$snapshot_dir/diagnostics/git_commit.txt" 2>/dev/null
python3 -V > "$snapshot_dir/diagnostics/python_version.txt"
pip freeze > "$snapshot_dir/diagnostics/pip_freeze.txt"
tree -a -I '__pycache__|.git|venv|.mypy_cache|.pytest_cache' > "$snapshot_dir/diagnostics/file_tree.txt"

echo "🗜️ Zipping everything..."
zip -r "saaf_os_diag_$timestamp.zip" "$snapshot_dir" > /dev/null
rm -rf "$snapshot_dir"

echo "✅ Snapshot complete: saaf_os_diag_$timestamp.zip"
