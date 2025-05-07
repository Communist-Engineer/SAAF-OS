#!/bin/bash

timestamp=$(date +%Y%m%d_%H%M%S)
snapshot_dir="saaf_os_snapshot_$timestamp"
mkdir -p "$snapshot_dir"/{tests,logs,models,memory,modules_subset,diagnostics}

echo "🔍 Collecting test files..."
cp -r tests/*.py "$snapshot_dir/tests/" 2>/dev/null

echo "🧪 Running pytest..."
pytest -v --tb=short | tee "$snapshot_dir/diagnostics/pytest.log"

echo "📁 Copying memory logs..."
cp memory/*.jsonl "$snapshot_dir/memory/" 2>/dev/null
cp memory/*.log "$snapshot_dir/logs/" 2>/dev/null

echo "🧠 Copying models..."
cp models/*.pt "$snapshot_dir/models/" 2>/dev/null

echo "🔬 Capturing key modules..."
MODULES=(planner.py rl_planner.py fwm.py contradiction_engine.py message_bus.py scenarios.py)
for m in "${MODULES[@]}"; do
  find modules -name "$m" -exec cp {} "$snapshot_dir/modules_subset/" \;
done

echo "🔢 Capturing system metadata..."
git rev-parse HEAD > "$snapshot_dir/diagnostics/git_commit.txt" 2>/dev/null
python3 -V > "$snapshot_dir/diagnostics/python_version.txt"
pip freeze > "$snapshot_dir/diagnostics/pip_freeze.txt"
tree -a -I '__pycache__|.git|.pytest_cache|*.zip|venv|.mypy_cache|__pycache__' > "$snapshot_dir/diagnostics/file_tree.txt"

echo "🗜️ Creating zip archive..."
zip -r "saaf_os_diagnostics_$timestamp.zip" "$snapshot_dir" > /dev/null
rm -rf "$snapshot_dir"

echo "✅ Snapshot complete: saaf_os_diagnostics_$timestamp.zip"
