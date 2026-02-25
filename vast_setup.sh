#!/bin/bash
set -e

# Parse arguments
BRANCH="main"
REPO_URL="https://github.com/trungkien04102002/spinet-v2.git"

while [[ $# -gt 0 ]]; do
    case $1 in
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--branch BRANCH_NAME]"
            echo "Default branch: test-run-project"
            exit 1
            ;;
    esac
done

echo "=== SpineNet Vast.ai Setup Script ==="
echo "Repository: $REPO_URL"
echo "Branch: $BRANCH"
echo ""

# Update system packages
echo "Updating system packages..."
apt-get update && apt-get install -y git wget

# Clone repository if not exists
REPO_DIR="spinet-v2"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository (branch: $BRANCH)..."
    git clone --branch $BRANCH $REPO_URL $REPO_DIR
else
    echo "Repository already exists, checking out branch $BRANCH..."
    cd $REPO_DIR
    git fetch origin
    git checkout $BRANCH
    git pull origin $BRANCH
    cd ..
fi

cd $REPO_DIR

# Create and activate virtual environment
echo "Creating virtual environment..."
if [ ! -d "spinenet-venv" ]; then
    python3 -m venv spinenet-venv
fi
source spinenet-venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set PYTHONPATH permanently
echo "Setting PYTHONPATH..."
SPINENET_PATH=$(pwd)
export PYTHONPATH=$PYTHONPATH:$SPINENET_PATH

# Add to bashrc if not already there
if ! grep -q "PYTHONPATH.*spinet-v2" ~/.bashrc; then
    echo "export PYTHONPATH=\$PYTHONPATH:$SPINENET_PATH" >> ~/.bashrc
fi

# Verify CUDA availability
echo ""
echo "=== Checking CUDA availability ==="
python3 << EOF
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device count: {torch.cuda.device_count()}')
    print(f'GPU device name: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available! SpineNet will run on CPU (much slower)')
EOF

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To run Python scripts:"
echo "  cd $REPO_DIR"
echo "  source spinenet-venv/bin/activate"
echo "  python test_spinenet.py"
echo ""
echo "The test script is configured to use CUDA. Develop with PyCharm/VSCode!"
