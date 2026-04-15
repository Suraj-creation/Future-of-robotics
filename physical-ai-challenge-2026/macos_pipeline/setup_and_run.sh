#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "[macos_pipeline] Working directory: $HERE"

# Create Python venv if missing
if [ ! -d "venv" ]; then
  echo "Creating Python venv..."
  python3 -m venv venv
fi

# Activate venv
# shellcheck disable=SC1091
source venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing Python packages from requirements.txt (this may take several minutes)..."
python -m pip install --no-cache-dir -r requirements.txt

# Create folders
mkdir -p sim/robots sim/worlds data models logs

# Touch placeholder for SO101 model location
if [ ! -f sim/robots/SO101.xml ]; then
  echo "# Place your SO101 MJCF/MJCF-equivalent here as sim/robots/SO101.xml" > sim/robots/SO101.xml
fi

# Run quick import/version test
echo "Running quick import test (will report missing packages)..."
python3 run_pipeline.py --test-imports

echo "Setup complete. To run the pipeline (once models are available):"
echo "  source venv/bin/activate"
echo "  python3 run_pipeline.py --start"

echo "If you need MuJoCo binaries or a license key, follow the README in macos_pipeline/README.md"
