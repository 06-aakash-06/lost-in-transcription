# Virtual Environment Setup

## Location
This project uses a Python virtual environment located in `./venv/`

## Activation

### For bash/zsh (macOS/Linux):
```bash
source venv/bin/activate
```

### For fish shell:
```bash
source venv/bin/activate.fish
```

### For Windows (Command Prompt):
```cmd
venv\Scripts\activate.bat
```

### For Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
```

## Installed Packages

Core data science & audio processing:
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **soundfile** - Audio file I/O
- **librosa** - Audio analysis and feature extraction
- **matplotlib** - Data visualization
- **scipy** - Scientific computing
- **scikit-learn** - Machine learning

Jupyter environment:
- **jupyter** - Jupyter notebook environment
- **ipykernel** - IPython kernel for Jupyter
- **notebook** - Jupyter web interface

All dependencies: see `requirements.txt`

## Using Multiple Notebooks

You can now create additional notebooks in this directory (e.g., `Analysis.ipynb`, `Preprocessing.ipynb`) and they will all use the same virtual environment.

### To run notebooks:
1. Activate the venv: `source venv/bin/activate`
2. Start Jupyter: `jupyter notebook`
3. Select the notebook you want to work on

## Updating Dependencies

If you need to add more packages:
```bash
source venv/bin/activate
pip install package_name
pip freeze > requirements.txt  # Update the requirements file
```

## IDE Setup

If using VSCode, select the interpreter at:
- `.venv/bin/python` or 
- In Command Palette: "Python: Select Interpreter" → choose the venv

If using JupyterLab/Notebook in IDE, make sure to select the kernel associated with this venv.
