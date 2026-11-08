# Build on Windows with: python -m PyInstaller --clean --noconfirm packaging/emilia.spec
# YOLO runs through the configured external Python; no CUDA/torch runtime is bundled.
from pathlib import Path
import sys

root = Path(SPECPATH).resolve().parent
sys.path.insert(0, str(root))
from build_support import resource_files

datas = [
    (str(source), str(Path("emilia_studio/resources") / source.relative_to(root).parent))
    for source in resource_files(root)
]
datas.append((str(root / "emilia_studio" / "inspection_worker.py"), "emilia_studio"))

analysis = Analysis(
    [str(root / "packaging" / "emilia_entry.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "torchvision", "ultralytics", "cv2"],
    noarchive=False,
)
archive = PYZ(analysis.pure)
exe = EXE(
    archive,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="emilia",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
