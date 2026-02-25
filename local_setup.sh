#!/bin/bash
set -e

echo "=== SpineNet Local Setup (MacOS) ==="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "spinenet-venv" ]; then
    python3 -m venv spinenet-venv
else
    echo "Virtual environment already exists, skipping..."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source spinenet-venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set PYTHONPATH
echo "Setting PYTHONPATH..."
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Virtual environment created at: $SCRIPT_DIR/spinenet-venv"
echo ""
echo "To activate the environment in the future, run:"
echo "  source spinenet-venv/bin/activate"
echo "  export PYTHONPATH=\$PYTHONPATH:$SCRIPT_DIR"
echo ""
echo "To run the test script:"
echo "  python test_spinenet.py"
echo ""
echo "⚠ Note: Your Mac will use CPU (no CUDA). The test script is already configured for CPU."
