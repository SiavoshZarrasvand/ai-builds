# Gemma-270M: Small Language Model from Scratch

This repository contains a Jupyter notebook implementation of the Gemma-270M small language model built from scratch using PyTorch. The model is designed to run on consumer hardware with an NVIDIA GPU (RTX 4070 with 8GB RAM minimum).

If any issues, contact [Zarrasvand.com](https://zarrasvand.com) for help.

**Note: This project requires NVIDIA GPU (RTX 4070 or higher) and is optimized for Windows.**

## Prerequisites

- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with 8GB+ VRAM (RTX 4070 or better recommended)
- **NVIDIA Drivers** ≥ 566.24 (check with `nvidia-smi`)
- **Git** for version control
- **PowerShell** (included with Windows)

## Getting Started

### 1. Clone the Repository

```powershell
git clone https://github.com/SiavashZarrasvand/ai-builds.git
cd ai-builds\Gemma-270M
```

### 2. Install Python 3.11.9

This project uses Python 3.11.9 as specified in `.python-version`. Install using winget:

```powershell
# Install Python 3.11.9 via winget
winget install --id Python.Python.3.11 --exact -s winget

# Verify installation
python --version  # Should output: Python 3.11.9
```

**Troubleshooting**: If `python` command shows Microsoft Store prompt:
- Remove Windows Python aliases: Go to Settings > Apps > Advanced app settings > App execution aliases
- Disable Python and Python3 aliases

### 3. Install UV Package Manager

UV is a fast Python package manager that handles virtual environments and dependencies:

```powershell
# Install UV
python -m pip install --upgrade pip uv

# Verify installation
uv --version
```

### 4. Set Up Project Environment

Create and activate a virtual environment, then install all dependencies:

```powershell
# Allow PowerShell script execution (if needed)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

# Create virtual environment
uv venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies from pyproject.toml
uv sync
```

**Note**: UV automatically manages dependencies using the `pyproject.toml` and `uv.lock` files for reproducible builds.

### 5. Install GPU-Enabled PyTorch

Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.7):

```powershell
# Install PyTorch with CUDA support
uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match --force-reinstall

# Verify GPU detection
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
```

### 6. Set Up Notebook Git Integration

Configure nbstripout to automatically clean notebook outputs before commits:

```powershell
# Install nbstripout git hooks
nbstripout --install

# Verify installation
nbstripout --status
```

This prevents large notebook outputs from being committed to version control.

### 7. Start JupyterLab

Launch JupyterLab to run the Gemma-270M notebook:

```powershell
# Start JupyterLab
jupyter lab

# Or run without activating virtual environment
uv run jupyter lab
```

Open your browser to `http://localhost:8888` and navigate to `Gemma_3_270_M_Small_Language_Model_Scratch_Final.ipynb`.

## Verify your setup

Run these commands to confirm CUDA and all dependencies are properly installed:

```powershell
# 1) Check NVIDIA driver and CUDA runtime
nvidia-smi

# 2) Verify PyTorch sees the GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# 3) Smoke-test key Python dependencies
python -c "import torch; import transformers; import datasets; import tiktoken; import matplotlib.pyplot as plt; import numpy as np; from tqdm import tqdm; print('All imports successful!')"

# 4) Confirm JupyterLab is installed
jupyter lab --version
```

Expected results:
- nvidia-smi prints GPU info and CUDA version
- CUDA available: True, GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- All imports successful!
- JupyterLab version prints (e.g., 4.4.6)

## Quick Start (Alternative)

If you already have the environment set up, you can quickly run the project:

```powershell
# Clone and navigate
git clone https://github.com/SiavashZarrasvand/ai-builds.git
cd ai-builds\Gemma-270M

# Activate environment and start
.venv\Scripts\Activate.ps1
jupyter lab
```

## UV Commands Reference

- `uv sync` - Install/update dependencies from lock file
- `uv add <package>` - Add a new dependency 
- `uv remove <package>` - Remove a dependency
- `uv run <command>` - Run command in the project environment
- `uv venv .venv` - Create a new virtual environment
- `uv pip install <package>` - Install packages directly into virtual environment

## Troubleshooting

### Common Issues

**Python not found after installation:**
- Restart PowerShell/Terminal
- Remove Windows Python aliases in Settings
- Add Python to PATH manually if needed

**CUDA not detected:**
- Verify NVIDIA drivers: `nvidia-smi`
- Ensure PyTorch CUDA version matches your system
- Try reinstalling PyTorch with explicit CUDA version

**nbstripout not working:**
```powershell
# Reinstall nbstripout
nbstripout --uninstall
nbstripout --install --attributes .gitattributes
```

**Virtual environment issues:**
```powershell
# Recreate virtual environment
Remove-Item -Recurse -Force .venv
uv venv .venv
.venv\Scripts\Activate.ps1
uv sync
```

## Project Structure

```
Gemma-270M/
├── .venv/                          # Virtual environment (created by uv)
├── .python-version                 # Python version specification
├── pyproject.toml                  # Project dependencies and configuration
├── uv.lock                         # Lock file with exact dependency versions
├── .gitignore                      # Git ignore rules
├── .gitattributes                  # Git attributes (nbstripout configuration)
├── README.md                       # This file
└── Gemma_3_270_M_Small_Language_Model_Scratch_Final.ipynb  # Main notebook
```

## Development Guidelines

**Version Control:**
- Never squash commits (repository policy)
- Always use `nbstripout` to clean notebook outputs
- Commit both code and markdown changes

**Environment Management:**
- Use `uv sync` to maintain consistent dependencies
- Update `pyproject.toml` for new dependencies
- Lock file (`uv.lock`) ensures reproducible builds

## Important Notes

**nbstripout** ensures only code and markdown changes are tracked in Git. Outputs and execution metadata are automatically stripped before each commit. **Everyone cloning this repo must set up nbstripout** to avoid accidental notebook changes in version control.

**Hardware Requirements:**
- Windows 10/11 (64-bit)
- NVIDIA GPU (RTX 4070 with 8GB RAM minimum)
- Python 3.11.9 (locked via `.python-version`)
- Git for version control
- At least 16GB system RAM recommended

---

**Support:** If you encounter issues, check the troubleshooting section above or contact [Zarrasvand.com](https://zarrasvand.com) for help.
