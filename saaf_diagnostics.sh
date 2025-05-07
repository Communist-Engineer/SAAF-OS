#!/bin/bash

# Create a temp diagnostics directory
SNAPSHOT_DIR="saaf_test_snapshot_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SNAPSHOT_DIR/tests"

echo "📦 Collecting test files..."
cp -r tests/*.py "$SNAPSHOT_DIR/tests/" 2>/dev/null

# Capture test metadata if exists
if [ -d ".pytest_cache" ]; then
    echo "🧪 Including pytest cache..."
    cp -r .pytest_cache "$SNAPSHOT_DIR/" 2>/dev/null
fi

# Run pytest and capture log
echo "📝 Running tests..."
pytest -v --tb=short | tee "$SNAPSHOT_DIR/test_results.log"

mkdir -p "$SNAPSHOT_DIR/memory"
cp memory/episodes.jsonl "$SNAPSHOT_DIR/memory/" 2>/dev/null

mkdir -p "$SNAPSHOT_DIR/models"
cp models/*.pt "$SNAPSHOT_DIR/models/" 2>/dev/null

mkdir -p "$SNAPSHOT_DIR/modules"
cp modules/scenarios.py "$SNAPSHOT_DIR/modules/" 2>/dev/null

git rev-parse HEAD > "$SNAPSHOT_DIR/git_commit.txt" 2>/dev/null

# Zip the entire test snapshot
ZIP_NAME="saaf_test_diagnostics.zip"
zip -r "$ZIP_NAME" "$SNAPSHOT_DIR" > /dev/null

# Cleanup
rm -rf "$SNAPSHOT_DIR"

echo "✅ Done. Upload $ZIP_NAME to ChatGPT."
