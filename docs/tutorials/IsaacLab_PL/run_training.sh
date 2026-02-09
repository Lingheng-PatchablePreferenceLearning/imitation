#!/bin/bash
# Launcher script for Preference Learning with IsaacLab Isaac-Ant-v0
#
# This script sets up the environment and runs the preference learning training.

set -e  # Exit on error

echo "=========================================="
echo "Preference Learning with Isaac-Ant-v0"
echo "=========================================="
echo ""

# Check if IsaacLab is installed
if ! python -c "import isaaclab" 2>/dev/null; then
    echo "ERROR: IsaacLab not found. Please ensure IsaacLab is installed."
    exit 1
fi

# Check if Imitation is installed
if ! python -c "import imitation" 2>/dev/null; then
    echo "ERROR: Imitation library not found. Please install it:"
    echo "  pip install imitation stable-baselines3"
    exit 1
fi

echo "✓ IsaacLab found"
echo "✓ Imitation library found"
echo ""

# Run the test script first
echo "Step 1: Testing wrapper..."
echo "-------------------------"
python test_wrapper.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Wrapper test failed. Please fix the issues before running training."
    exit 1
fi

echo ""
echo "✓ Wrapper test passed!"
echo ""

# Ask user if they want to continue with training
read -p "Continue with full preference learning training? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Run the main training script
echo ""
echo "Step 2: Running preference learning training..."
echo "-----------------------------------------------"
python train_preference_learning_ant.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Training failed. Check the error messages above."
    echo "=========================================="
    exit 1
fi
