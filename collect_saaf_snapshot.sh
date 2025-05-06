#!/bin/bash

# Set snapshot directory
SNAPSHOT_DIR="saaf_snapshot_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SNAPSHOT_DIR"

echo "📁 Creating snapshot directory: $SNAPSHOT_DIR"

# Copy relevant directories if they exist
for dir in modules scripts config tests; do
  if [ -d "$dir" ]; then
    echo "📦 Copying $dir/"
    cp -r "$dir" "$SNAPSHOT_DIR/"
  fi
done

# Save file tree (excluding virtualenvs, git, pycache)
echo "📄 Saving file tree..."
tree -I '__pycache__|.git|venv|env|.mypy_cache|.pytest_cache' > "$SNAPSHOT_DIR/file_tree.txt"

# Optionally copy latest log file if exists
LOG_FILE=$(find . -name "*.log" | head -n 1)
if [ -n "$LOG_FILE" ]; then
  echo "📜 Including log file: $LOG_FILE"
  cp "$LOG_FILE" "$SNAPSHOT_DIR/"
fi

# Compress snapshot
ZIP_NAME="saaf-os_snapshot.zip"
echo "📦 Zipping to $ZIP_NAME"
zip -r "$ZIP_NAME" "$SNAPSHOT_DIR" > /dev/null

# Cleanup directory
rm -rf "$SNAPSHOT_DIR"

echo "✅ Done. Upload $ZIP_NAME to ChatGPT."
