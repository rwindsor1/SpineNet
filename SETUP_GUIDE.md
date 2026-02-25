# SpineNet Setup Guide - Python Development

## Quick Setup for Local Machine (MacOS)

### Automated Setup
```bash
cd /Users/kienha/spinet-v2
./local_setup.sh
```

### Manual Setup
```bash
cd /Users/kienha/spinet-v2

# Create virtual environment
python3 -m venv spinenet-venv
source spinenet-venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/Users/kienha/spinet-v2
```

### Run Test Script
```bash
source spinenet-venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/Users/kienha/spinet-v2
python test_spinenet.py
```

**Note**: Your Mac will run on CPU (no CUDA). The test script is already configured for CPU mode.

---

## Setup for Vast.ai GPU Machine

### 1. Choose a Vast.ai Instance
- Select an instance with:
  - CUDA-enabled GPU (e.g., RTX 3090, RTX 4090, A6000)
  - At least 16GB RAM
  - PyTorch template or Ubuntu 20.04/22.04

### 2. SSH into Your Vast.ai Instance
```bash
ssh -p [PORT] root@[IP_ADDRESS]
```

### 3. Upload and Run Setup Script
```bash
# From your local machine, upload the setup script
scp -P [PORT] vast_setup.sh root@[IP_ADDRESS]:~/

# SSH into the machine
ssh -p [PORT] root@[IP_ADDRESS]

# Run the setup script (uses main branch by default)
chmod +x vast_setup.sh
./vast_setup.sh

# Or specify a different branch
./vast_setup.sh --branch test-run-project
./vast_setup.sh --branch your-feature-branch
```

### 4. Run Python Scripts
```bash
cd spinet-v2
source spinenet-venv/bin/activate
python test_spinenet.py
```

---

## Development with PyCharm/VSCode

### VSCode Setup
1. Open project folder: `/Users/kienha/spinet-v2` (local) or `~/SpineNetV2` (vast.ai)
2. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Linux)
3. Type "Python: Select Interpreter"
4. Choose `./spinenet-venv/bin/python`

### PyCharm Setup
1. Open project folder
2. Go to: Settings → Project → Python Interpreter
3. Add interpreter → Existing environment
4. Select: `spinenet-venv/bin/python`

### Important Note
Make sure PYTHONPATH includes the project root:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

---

## Test Script Usage

The `test_spinenet.py` script demonstrates the core SpineNet pipeline:

```python
# Local Mac (CPU)
device = 'cpu'

# Vast.ai GPU (CUDA)
device = 'cuda:0'
```

It will:
1. Download example T2 lumbar MRI scan (~2.5MB)
2. Download pre-trained model weights (~2-3GB, first run only)
3. Detect and label vertebrae (T11-S1)
4. Extract intervertebral discs (IVDs)
5. Grade IVDs for various spinal conditions
6. Save results to `results/test_results.csv`

---

## Performance Expectations

### Local Mac (CPU)
- Initial setup: ~5 minutes
- Model weight download: ~5 minutes (first run only)
- Single scan processing: ~5-10 minutes
- Good for: Testing, development, small batches

### Vast.ai GPU (CUDA)
- Initial setup: ~5 minutes
- Model weight download: ~3 minutes (first run only)
- Single scan processing: ~30-60 seconds
- Good for: Production, batch processing, research

---

## Troubleshooting

### Import Errors
Make sure PYTHONPATH is set:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Or add this to your Python scripts:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
```

### CUDA Out of Memory
If you get CUDA OOM errors on vast.ai:
- Try a GPU with more VRAM (24GB+)
- Process scans one at a time
- Check for other processes using GPU: `nvidia-smi`

### Missing matplotlib Error
```bash
pip install matplotlib
```

### Missing DICOM Metadata
The example scans have missing metadata that's handled by `overwrite_dict` in the test script. For your own scans, ensure they have proper DICOM headers or add appropriate overwrites.

---

## Next Steps

1. Run `test_spinenet.py` to verify setup
2. Read the test script to understand the API
3. Develop your own scripts using PyCharm/VSCode
4. Use `test_spinenet.py` as a template for your own DICOM processing

For questions about the SpineNet model or grading schemes, see:
- Paper: http://zeus.robots.ox.ac.uk/spinenet2/
- GitHub: https://github.com/rwindsor1/SpineNetV2
