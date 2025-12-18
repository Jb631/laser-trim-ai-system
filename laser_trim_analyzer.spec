# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Laser Trim Analyzer

Build with: pyinstaller laser_trim_analyzer.spec --clean
"""

import sys
from pathlib import Path

# Get version from pyproject.toml
import tomllib
with open("pyproject.toml", "rb") as f:
    version = tomllib.load(f)["project"]["version"]

block_cipher = None

# Collect data files
datas = [
    ('src/laser_trim_analyzer', 'laser_trim_analyzer'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'laser_trim_analyzer',
    'laser_trim_analyzer.app',
    'laser_trim_analyzer.config',
    'laser_trim_analyzer.core',
    'laser_trim_analyzer.core.parser',
    'laser_trim_analyzer.core.processor',
    'laser_trim_analyzer.core.models',
    'laser_trim_analyzer.database',
    'laser_trim_analyzer.database.manager',
    'laser_trim_analyzer.database.models',
    'laser_trim_analyzer.gui',
    'laser_trim_analyzer.gui.app',
    'laser_trim_analyzer.gui.pages',
    'laser_trim_analyzer.gui.pages.dashboard',
    'laser_trim_analyzer.gui.pages.process',
    'laser_trim_analyzer.gui.pages.analyze',
    'laser_trim_analyzer.gui.pages.trends',
    'laser_trim_analyzer.gui.pages.settings',
    'laser_trim_analyzer.gui.widgets',
    'laser_trim_analyzer.gui.widgets.chart',
    'laser_trim_analyzer.ml',
    'laser_trim_analyzer.ml.threshold',
    'laser_trim_analyzer.export',
    'laser_trim_analyzer.export.excel',
    # Dependencies
    'customtkinter',
    'tkinterdnd2',
    'PIL',
    'PIL._tkinter_finder',
    'openpyxl',
    'sqlalchemy',
    'sqlalchemy.dialects.sqlite',
    'pydantic',
    'pydantic_settings',
    'numpy',
    'matplotlib',
    'matplotlib.backends.backend_tkagg',
    'scipy',
    'scipy.stats',
    'sklearn',
    'sklearn.ensemble',
    'sklearn.preprocessing',
    'joblib',
    'yaml',
    'platformdirs',
]

a = Analysis(
    ['src/laser_trim_analyzer/__main__.py'],
    pathex=['src'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Testing
        'test',
        'tests',
        'pytest',
        'unittest',
        # Heavy ML libraries NOT used by this app
        'tensorflow',
        'tensorflow_core',
        'tensorflow_estimator',
        'tensorboard',
        'torch',
        'torchvision',
        'torchaudio',
        'transformers',
        'huggingface_hub',
        'tokenizers',
        'safetensors',
        # Data libraries not needed
        'pyarrow',
        'pandas',
        'h5py',
        # Other unused
        'IPython',
        'ipykernel',
        'jupyter',
        'notebook',
        'jedi',
        'Pythonwin',
        'win32com',
        'zmq',
        'grpc',
        'grpcio',
        'cryptography',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out unnecessary files to reduce size
exclude_patterns = [
    'tests', 'test_', '_test.py', '__pycache__', '.pyc',
    'tensorflow', 'torch', 'transformers', 'huggingface',
    'pyarrow', 'pandas', 'h5py', 'grpc', 'Pythonwin',
]
a.datas = [d for d in a.datas if not any(x in d[0] for x in exclude_patterns)]

# Also filter binaries
a.binaries = [b for b in a.binaries if not any(x in b[0].lower() for x in [
    'tensorflow', 'torch', 'libtorch', 'transformers', 'pyarrow',
    'pandas', 'h5py', 'grpc', 'zmq', 'crypto'
])]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LaserTrimAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path if you have one: icon='assets/icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=f'LaserTrimAnalyzer-v{version}',
)
