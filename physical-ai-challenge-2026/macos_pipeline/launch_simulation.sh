#!/bin/bash
# Launch script for Autonomous Pick-and-Place Simulation
# This script sets up the environment and runs the SO101 MuJoCo simulation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  SO101 Autonomous Pick & Place Setup  ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for Python
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}FAILED${NC}"
    echo "Python is not installed. Please install Python 3.8+."
    exit 1
fi
echo -e "${GREEN}OK${NC} ($($PYTHON --version))"

# Check for virtual environment
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo -n "Activating virtual environment... "
    source "$SCRIPT_DIR/venv/bin/activate"
    echo -e "${GREEN}OK${NC}"
fi

# Check required packages
echo ""
echo "Checking required packages..."
REQUIRED_PKGS=("mujoco" "numpy" "ultralytics")
MISSING_PKGS=()

for pkg in "${REQUIRED_PKGS[@]}"; do
    echo -n "  $pkg... "
    if $PYTHON -c "import $pkg" 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}MISSING${NC}"
        MISSING_PKGS+=($pkg)
    fi
done

if [ ${#MISSING_PKGS[@]} -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}Installing missing packages...${NC}"
    pip install "${MISSING_PKGS[@]}"
fi

# Check for SO101 model
echo ""
echo -n "Checking SO101 model... "
if [ -f "$SCRIPT_DIR/third_party/SO-ARM100/Simulation/SO101/so101_new_calib.xml" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}NOT FOUND${NC}"
    echo "Please ensure SO101 model is at:"
    echo "  third_party/SO-ARM100/Simulation/SO101/so101_new_calib.xml"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Environment Ready!                   ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Parse arguments
CYCLES=10
HEADLESS=""
YOLO_MODEL=""
ACT_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cycles)
            CYCLES="$2"
            shift 2
            ;;
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        --yolo-model)
            YOLO_MODEL="--yolo-model $2"
            shift 2
            ;;
        --act-model)
            ACT_MODEL="--act-model $2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Running autonomous simulation..."
echo "  Cycles: $CYCLES"
echo "  Mode: $(if [ -n "$HEADLESS" ]; then echo "Headless"; else echo "Visual"; fi)"
echo ""

# Run the simulation
$PYTHON "$SCRIPT_DIR/run_autonomous.py" \
    --cycles "$CYCLES" \
    $HEADLESS \
    $YOLO_MODEL \
    $ACT_MODEL

echo ""
echo -e "${GREEN}Simulation complete!${NC}"
