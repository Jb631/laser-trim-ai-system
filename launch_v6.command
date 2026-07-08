#!/bin/bash
cd "$(dirname "$0")"
echo "Starting Laser Trim Analyzer V6..."
# Use the project's framework .venv directly so the GUI runs under a
# visible Python.app bundle (uv's standalone python has no grantable identity).
if [ -x ".venv/bin/python" ]; then
    .venv/bin/python -m src --v6 2>&1 | tee /tmp/laser_app_v6_log.txt
elif command -v uv &>/dev/null; then
    uv run python -m src --v6 2>&1 | tee /tmp/laser_app_v6_log.txt
else
    python3 -m src --v6 2>&1 | tee /tmp/laser_app_v6_log.txt
fi
