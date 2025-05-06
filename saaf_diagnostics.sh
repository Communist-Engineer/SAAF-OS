#!/bin/bash

echo "🧾 SAAF-OS Diagnostic Report (macOS)"
echo "==================================="

echo -e "\n📁 Directory Tree (depth 2)"
tree -L 2 || echo "tree not found. Install with: brew install tree"

echo -e "\n📄 Python Modules in modules/"
find modules -name '*.py' | sort

echo -e "\n🛠 TODOs / Stubs in Modules"
grep -Rni 'TODO' modules/ || echo "No TODOs found"

echo -e "\n🧪 Pytest with Coverage"
pytest --cov=modules --cov-report=term

echo -e "\n📊 Coverage Detail (missing lines)"
coverage report -m || echo "Install with: pip install coverage"

echo -e "\n📂 RSI Patch Signature Function"
grep -A5 'def _verify_patch_signature' modules/rsi/engine.py

echo -e "\n🔐 Key Existence Check"
ls -l data/keys/

echo -e "\n🕵️ Git Ignore Check"
git check-ignore data/keys/* || echo "Key dir not ignored!"

echo -e "\n📦 Git Status + Diff"
git status
git diff --stat HEAD~1

echo -e "\n🧾 Recent Commit File List"
git diff --name-status HEAD~1

echo -e "\n🧪 Scenario Result JSON (if present)"
cat results/results.json 2>/dev/null | jq . || echo "No results.json or jq not installed"

echo "✅ Done"
