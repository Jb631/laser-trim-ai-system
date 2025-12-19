# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Laser Trim Analyzer

Build with: pyinstaller laser_trim_analyzer.spec --clean
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# Get version from pyproject.toml
import tomllib
with open("pyproject.toml", "rb") as f:
    version = tomllib.load(f)["project"]["version"]

block_cipher = None

# Only collect what we actually use
# scipy - needed for signal processing (butter, filtfilt, optimize, stats)
scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all('scipy')

# customtkinter - GUI framework (needs theme files)
ctk_datas, ctk_binaries, ctk_hiddenimports = collect_all('customtkinter')

# sklearn - needed for RandomForest ML threshold optimization
sklearn_datas, sklearn_binaries, sklearn_hiddenimports = collect_all('sklearn')

# Collect data files
datas = [
    ('src/laser_trim_analyzer', 'laser_trim_analyzer'),
]
datas += scipy_datas
datas += ctk_datas
datas += sklearn_datas

# Hidden imports - only what the app actually uses
hiddenimports = [
    # App modules
    'laser_trim_analyzer',
    'laser_trim_analyzer.app',
    'laser_trim_analyzer.config',
    'laser_trim_analyzer.core',
    'laser_trim_analyzer.core.parser',
    'laser_trim_analyzer.core.processor',
    'laser_trim_analyzer.core.models',
    'laser_trim_analyzer.core.analyzer',
    'laser_trim_analyzer.database',
    'laser_trim_analyzer.database.manager',
    'laser_trim_analyzer.database.models',
    'laser_trim_analyzer.gui',
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
    'laser_trim_analyzer.ml.drift',
    'laser_trim_analyzer.export',
    'laser_trim_analyzer.export.excel',
    'laser_trim_analyzer.utils',
    'laser_trim_analyzer.utils.constants',
    # Core dependencies actually used
    'customtkinter',
    'PIL',
    'PIL._tkinter_finder',
    'PIL.Image',
    'PIL.ImageTk',
    'openpyxl',
    'openpyxl.workbook',
    'openpyxl.styles',
    'openpyxl.utils',
    'sqlalchemy',
    'sqlalchemy.dialects.sqlite',
    'sqlalchemy.orm',
    'sqlalchemy.ext.declarative',
    'sqlalchemy.pool',
    'sqlalchemy.exc',
    'pydantic',
    'pydantic.dataclasses',
    'numpy',
    'numpy.core._methods',
    'numpy.lib.format',
    'pandas',
    'pandas.core',
    'pandas._libs',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.figure',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_agg',
    'scipy',
    'scipy.stats',
    'scipy.signal',
    'scipy.optimize',
    'sklearn',
    'sklearn.ensemble',
    'sklearn.model_selection',
    'sklearn.metrics',
    'sklearn.preprocessing',
    'joblib',
    'yaml',
    'pickle',
    # Tkinter
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    '_tkinter',
]

a = Analysis(
    ['src/laser_trim_analyzer/__main__.py'],
    pathex=['src'],
    binaries=scipy_binaries + ctk_binaries + sklearn_binaries,
    datas=datas,
    hiddenimports=hiddenimports + scipy_hiddenimports + ctk_hiddenimports + sklearn_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Testing (keep unittest - sklearn needs it)
        'test', 'tests', 'pytest',
        # Heavy ML libraries NOT used
        'tensorflow', 'tensorflow_core', 'tensorflow_estimator', 'tensorboard',
        'torch', 'torchvision', 'torchaudio',
        'transformers', 'huggingface_hub', 'tokenizers', 'safetensors',
        # Data libraries not used
        'pyarrow', 'h5py', 'zarr',
        # Not used
        'IPython', 'ipykernel', 'jupyter', 'notebook', 'jedi',
        'Pythonwin', 'win32com',
        'zmq', 'grpc', 'grpcio',
        'cryptography',
        'httpx', 'requests',
        'rich', 'click',
        'watchdog', 'diskcache',
        'statsmodels', 'seaborn',
        'tkinterdnd2',  # Not actually used
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out unnecessary files to reduce size
exclude_patterns = [
    'tests', 'test_', '_test.py', '__pycache__',
    'tensorflow', 'torch', 'transformers', 'huggingface',
    'pyarrow', 'h5py', 'grpc', 'Pythonwin',
]
a.datas = [d for d in a.datas if not any(x in d[0] for x in exclude_patterns)]

# Filter binaries
a.binaries = [b for b in a.binaries if not any(x in b[0].lower() for x in [
    'tensorflow', 'torch', 'libtorch', 'transformers', 'pyarrow',
    'h5py', 'grpc', 'zmq', 'crypto', 'libcrypto', 'libssl'
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
    console=False,  # No console for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
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
