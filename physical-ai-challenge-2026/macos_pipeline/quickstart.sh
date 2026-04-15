#!/bin/bash
# Quick start script for Autonomous Pick-and-Place

cd "$(dirname "$0")"

echo "SO101 Autonomous Pick-and-Place Simulation"
echo "=========================================="
echo ""
echo "Running autonomous simulation with 5 cycles..."
echo ""

python3 run_autonomous.py --cycles 5

echo ""
echo "Simulation complete!"
echo ""
echo "For more options, use:"
echo "  python3 run_autonomous.py --help"
