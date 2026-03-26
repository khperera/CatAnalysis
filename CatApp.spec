# PyInstaller spec for CatApp
# Build with:  pyinstaller CatApp.spec

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['CatApp.py'],
    pathex=[str(Path('.').resolve())],
    binaries=[],
    datas=[
        # Bundle the trained model weights
        ('cat_classifier_mobilenet_v3.pth', '.'),
        # Bundle YOLO model weights (downloaded on first run if missing,
        # but including it avoids the network fetch)
        ('yolo11m-seg.pt', '.'),
        # Include the app package
        ('app', 'app'),
    ],
    hiddenimports=[
        'ultralytics',
        'ultralytics.models',
        'ultralytics.models.yolo',
        'ultralytics.utils',
        'PIL._tkinter_finder',
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'tkinter',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude hardware-control modules not needed in the new app
        'pyftdi',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CatDetector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # no terminal window on launch
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CatDetector',
)
