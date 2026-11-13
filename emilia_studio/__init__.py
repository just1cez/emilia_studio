"""Emilia Studio — Unified Declarative SDXL Quality Pipeline, Inspector, Compiler, TUI, and Ledger."""

from ._version import __version__

from .compiler import BridgeError, CompiledPromptSpec, OllamaPromptCompiler
from .config import BUILTIN_SCENE_PRESETS, StudioConfig, get_user_config_path, load_config
from .engine import EmiliaRenderEngine, StageResult
from .inspector import InspectionReport, QualityInspector
from .ledger import ExperimentLedger
from .tui import EMILIA_SLANT_ASCII, ProgressBar, download_file_with_progress, print_banner, render_emilia_banner, render_table
from .workflows import EmiliaWorkflowBuilder, WorkflowSpec

__all__ = [
    "__version__",
    "BUILTIN_SCENE_PRESETS",
    "BridgeError",
    "CompiledPromptSpec",
    "EMILIA_SLANT_ASCII",
    "EmiliaRenderEngine",
    "EmiliaWorkflowBuilder",
    "ExperimentLedger",
    "InspectionReport",
    "OllamaPromptCompiler",
    "ProgressBar",
    "QualityInspector",
    "StageResult",
    "StudioConfig",
    "WorkflowSpec",
    "download_file_with_progress",
    "get_user_config_path",
    "load_config",
    "print_banner",
    "render_emilia_banner",
    "render_table",
]
