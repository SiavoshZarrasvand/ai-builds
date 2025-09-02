# ai-builds

This repository contains Jupyter notebooks for language model experiments. All models here are small, meaning an RTX 4070 with 8GB of RAM should be able to run the notebooks.
If any issues, contact [Zarrasvand.com](https://zarrasvand.com) for help.

**Note: This project requires NVIDIA GPU (RTX 4070 or higher) and only runs on Windows.**

## Getting Started (Windows Only)

### 1. Clone the repository

```cmd
git clone https://github.com/SiavashZarrasvand/ai-builds.git
cd ai-builds
```

### 2. Install Python 3.11.9

This project uses Python 3.11.9 as specified in `.python-version`. Install it using:

```cmd
rem If using pyenv-win
pyenv install 3.11.9

rem Or download Python 3.11.9 from https://python.org and install manually
```

### 3. Install UV (Python package manager)

```cmd
pip install uv
rem Or use the uv installer from https://docs.astral.sh/uv/getting-started/installation/
```

### 4. Install dependencies with UV

UV will automatically create a virtual environment and install all dependencies:

```cmd
uv sync
```

If there's no `pyproject.toml` yet, initialize the project:
```cmd
uv init
uv add jupyter nbstripout torch transformers datasets
```

### 5. Activate the virtual environment

```cmd
.venv\Scripts\activate
```

### 6. Set up nbstripout to clean notebook metadata on commit

After running `uv sync`, nbstripout should already be installed. Set it up:

```cmd
rem Set up nbstripout for this repository
nbstripout --install

rem Verify it's working
nbstripout --status
```

### 7. Start Jupyter

```cmd
jupyter lab
```

Or without activating the virtual environment:
```cmd
uv run jupyter lab
```

## UV Commands Reference

- `uv sync` - Install/update dependencies from lock file
- `uv add <package>` - Add a new dependency 
- `uv remove <package>` - Remove a dependency
- `uv run <command>` - Run command in the project environment

## Important Notes

**nbstripout** ensures only code and markdown changes are tracked in Git. Outputs and execution metadata are automatically stripped before each commit. **Everyone cloning this repo must set up nbstripout** to avoid accidental notebook changes in version control.

If you encounter any issues with nbstripout, you can reinstall it:
```cmd
nbstripout --uninstall
nbstripout --install
```

**Requirements:**
- Windows OS
- NVIDIA GPU (RTX 4070 with 8GB RAM minimum)
- Python 3.11.9 (locked via `.python-version`)
- Git

---

If you have any issues, make sure you have Python 3.11.9, NVIDIA drivers, and Git installed.
