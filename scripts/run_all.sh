#!/bin/bash
set -e

echo "🚀 Starting full pipeline..."
python3 scripts/train.py
python3 scripts/merge_and_convert.py
echo "✅ All done! Check /workspace/models"
