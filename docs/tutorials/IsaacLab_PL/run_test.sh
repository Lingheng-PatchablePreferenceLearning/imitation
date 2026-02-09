#!/bin/bash
# Run the test wrapper using IsaacLab's Python environment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ISAACLAB_DIR="/home/lingheng/IsaacLab_Stack_3/hri-ppl/IsaacLab"

echo "Running test_wrapper.py with IsaacLab Python environment..."
cd "$SCRIPT_DIR"
"$ISAACLAB_DIR/isaaclab.sh" -p test_wrapper.py
