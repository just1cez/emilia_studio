"""Build-time resource staging shared by wheel and Windows executable builds."""

from __future__ import annotations

import shutil
from pathlib import Path

from setuptools.command.build_py import build_py


RESOURCE_PATTERNS = (
    "models_manifest.json",
    "assets/**/*.png",
    "workflows/api/*.json",
    "workflows/visual/*.json",
    "patches/*.patch",
    "patches/emilia_runtime_patches/**/*.py",
)


def resource_files(root: Path) -> list[Path]:
    """Return canonical, sorted resources from either a checkout or an sdist."""
    files = {path for pattern in RESOURCE_PATTERNS for path in root.glob(pattern) if path.is_file()}
    for required in ("models_manifest.json", "assets", "workflows/api", "workflows/visual", "patches/emilia_runtime_patches"):
        if not (root / required).exists():
            raise FileNotFoundError(f"Missing distribution resource: {root / required}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


class ResourceBuildPy(build_py):
    """Stage canonical resources without modifying or duplicating tracked files."""

    def run(self) -> None:
        super().run()
        root = Path(__file__).resolve().parent
        destination = Path(self.build_lib) / "emilia_studio" / "resources"
        sources = resource_files(root)
        if destination.exists():
            shutil.rmtree(destination)
        for source in sources:
            target = destination / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        outputs = super().get_outputs(include_bytecode)
        root = Path(__file__).resolve().parent
        destination = Path(self.build_lib) / "emilia_studio" / "resources"
        outputs.extend(str(destination / source.relative_to(root)) for source in resource_files(root))
        return outputs
