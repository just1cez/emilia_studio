"""PyInstaller entry point; keep package-relative imports in the real CLI."""

from multiprocessing import freeze_support

from emilia_studio.cli import main


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
