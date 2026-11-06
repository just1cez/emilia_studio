"""Configuration and canonical constants for Emilia Studio."""

from __future__ import annotations

import os
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Canonical Emilia identity and prompt constants
EMILIA_CANONICAL: str = (
    r"1girl, emilia \(re:zero\), re:zero kara hajimeru isekai seikatsu, safe, solo, "
    "silver hair, purple eyes, braid, hair flower, x hair ornament, "
    "white dress, detached sleeves, green gem"
)
EMILIA_REFERENCE: str = "https://re-zero-anime.jp/tv/character/"
EMILIA_ALIASES: tuple[str, ...] = ("爱蜜莉雅", "艾米莉娅", "エミリア", "emilia")

QUALITY_TAGS: str = "masterpiece, high score, great score, absurdres"
STYLE_HINTS: dict[str, str] = {
    "anime": "Japanese anime key visual, clean expressive line art, soft cel shading, polished 2D animation finish",
    "illustration": "polished fantasy illustration, refined line art, luminous painterly details",
    "cinematic": "cinematic character artwork, dramatic composition, atmospheric lighting, detailed rendering",
}

# High-res quality production default prompts (1152x1728 -> 2304x3456)
DEFAULT_HIGHRES_POSITIVE_PROMPT: str = (
    r"masterpiece, best quality, anime illustration, emilia \(re:zero\), 1girl, solo, "
    "silver-white long hair, crown braid, straight bangs, matching violet eyes, "
    "pointed elf ears, white flower hair ornament, x-shaped hair clip, purple ribbons, "
    "white and lavender dress with purple trim, green gemstone brooch, detached bell sleeves, "
    "white knee-high boots, full body, both boots fully visible, standing on a garden path, "
    "hands relaxed, gentle smile, looking at viewer, balanced anatomy, flower garden, "
    "blooming flowers, leafy trees, stone archway, blue sky, warm soft sunlight, "
    "coherent painted background, layered cel shading, delicate clean lineart, subtle fabric folds"
)

DEFAULT_HIGHRES_NEGATIVE_PROMPT: str = (
    "worst quality, low quality, unfinished, monochrome, sketch, lineart background, "
    "wireframe, scribbles, messy background, overexposed, flat lighting, blurry, "
    "bad anatomy, bad hands, fused fingers, extra fingers, missing fingers, extra limbs, "
    "fused legs, barefoot, deformed feet, cropped feet, cropped head, multiple girls, "
    "heterochromia, blue eyes, red eyes, pink eyes, text, watermark, signature, photo, realistic, 3d, nsfw"
)

# High-res finish FaceDetailer prompts
DEFAULT_FACE_DETAILER_POSITIVE_PROMPT: str = (
    r"emilia \(re:zero\), anime face, matching violet irises, symmetrical eyes, "
    "natural eye highlights, gentle smile, silver-white hair, straight bangs, "
    "crown braid, white flower ornament, x-shaped hair clip, pointed elf ears, "
    "delicate clean lineart, layered cel shading, preserve original expression and identity"
)

DEFAULT_FACE_DETAILER_NEGATIVE_PROMPT: str = (
    "blurry face, soft eyes, asymmetrical eyes, crossed eyes, red eyes, blue eyes, "
    "pink eyes, heterochromia, malformed pupils, extra eyes, missing eyes, distorted face, "
    "deformed mouth, old face, realistic, photo, 3d, over-smoothed plastic face, text, watermark"
)

# High-res finish HandDetailer prompts
DEFAULT_HAND_DETAILER_POSITIVE_PROMPT: str = (
    "anime hands, natural relaxed fingers, five fingers on each hand, correct finger joints "
    "and wrists, pale skin, preserve the original pose and sleeve boundaries, "
    "delicate clean lineart, soft cel shading"
)

DEFAULT_HAND_DETAILER_NEGATIVE_PROMPT: str = (
    "malformed hand, extra fingers, missing fingers, fused fingers, twisted wrist, "
    "duplicated hand, extra arm, deformed sleeve, blurry, realistic, photo, text, watermark"
)

# Eye repair fallback prompts
DEFAULT_EYE_REPAIR_BASE_POSITIVE_PROMPT: str = (
    r"1girl, solo, rezero emilia, emilia \(re:zero\), silver-white hair, straight bangs, "
    "crown braid, violet eyes, pointed elf ears, white flower hair ornament, "
    "four-petal x-shaped hair clip, purple ribbon, white and lavender dress, white cape, "
    "detached sleeves, purple trim, green gemstone brooch, gentle smile, anime key visual, "
    "full body, standing, both arms visible, hands visible, centered composition, "
    "flower garden, soft daylight, clean anime lineart, refined cel shading, masterpiece, best quality"
)

DEFAULT_EYE_REPAIR_BASE_NEGATIVE_PROMPT: str = (
    "worst quality, low quality, lowres, blurry, out of focus, bad anatomy, bad hands, "
    "malformed hands, fused fingers, extra fingers, missing fingers, extra arms, extra legs, "
    "extra limbs, multiple girls, asymmetrical eyes, crossed eyes, red eyes, blue eyes, "
    "heterochromia, deformed face, deformed eyes, distorted pupils, text, watermark, "
    "signature, logo, photo, realistic, 3d, chibi, child, mature woman, nsfw, nude"
)

DEFAULT_EYE_REPAIR_POSITIVE_PROMPT: str = (
    "anime face, Emilia Re:Zero, ((both eyes same violet color:1.4)), ((violet eyes:1.5)), "
    "symmetrical anime eyes, identical iris color, clear round irises, natural highlights, "
    "straight bangs, crown braid, white flower hair ornament, x-shaped hair clip, "
    "pointed elf ears, gentle warm smile, clean lineart, preserve silver-white hair and identity"
)

DEFAULT_EYE_REPAIR_NEGATIVE_PROMPT: str = (
    "blue eyes, pink eyes, red eyes, mismatched eyes, heterochromia, asymmetrical eyes, "
    "crossed eyes, malformed pupils, extra eyes, missing eyes, distorted face, deformed mouth, "
    "blurry face, realistic, photo, 3d, plastic face, text, watermark"
)

# Lightweight LoRA native prompts (832x1216)
DEFAULT_LORA_NATIVE_POSITIVE_PROMPT: str = (
    r"1girl, solo, rezero emilia, emilia \(re:zero\), re:zero kara hajimeru isekai seikatsu, "
    "anime key visual, official character design, silver-white hair, long hair, straight bangs, "
    "crown braid, violet eyes, pointed elf ears, white flower hair ornament, "
    "four-petal x-shaped hair clip, purple ribbon, white and lavender dress, white cape, "
    "detached sleeves, purple trim, green gemstone brooch, gentle smile, looking at viewer, "
    "full body, standing, both arms visible, hands visible, relaxed natural pose, "
    "balanced proportions, centered composition, flower garden, stone balustrade, "
    "soft daylight, pale blue sky, clean anime lineart, refined cel shading, soft colors, "
    "masterpiece, best quality"
)

DEFAULT_LORA_NATIVE_NEGATIVE_PROMPT: str = (
    "worst quality, low quality, lowres, blurry, out of focus, bad anatomy, bad hands, "
    "malformed hands, fused fingers, extra fingers, missing fingers, extra arms, extra legs, "
    "extra limbs, disconnected limbs, duplicated body, multiple girls, cropped head, "
    "cropped feet, cut off, close-up, text, watermark, signature, logo, jpeg artifacts, "
    "photo, realistic, 3d, chibi, child, mature woman, nsfw, nude, overly sexualized, "
    "giant breasts, asymmetrical eyes, crossed eyes, red eyes, blue eyes, heterochromia, "
    "deformed clothing, tangled hair"
)

# Safe LoRA 832x1216 baseline prompts
DEFAULT_LORA_SAFE_POSITIVE_PROMPT: str = (
    r"1girl, solo, rezero emilia, emilia \(re:zero\), re:zero kara hajimeru isekai seikatsu, "
    "anime key visual, official character design, silver-white hair, long hair, straight bangs, "
    "crown braid, violet eyes, pointed elf ears, white flower hair ornament, "
    "four-petal x-shaped hair clip, purple ribbon, white and lavender dress, white cape, "
    "detached sleeves, purple trim, green gemstone brooch, gentle smile, looking at viewer, "
    "full body, character focus, medium-long shot, subject fills the frame, feet visible, "
    "arms relaxed at sides, hands visible, simple standing pose, balanced body proportions, "
    "centered composition, flower garden, stone archway, soft daylight, pale blue sky, "
    "clean anime lineart, refined cel shading, soft colors, masterpiece, best quality"
)

DEFAULT_LORA_SAFE_NEGATIVE_PROMPT: str = (
    "worst quality, low quality, lowres, blurry, out of focus, bad anatomy, bad hands, "
    "malformed hands, fused fingers, extra fingers, missing fingers, extra arms, extra legs, "
    "extra limbs, disconnected limbs, duplicated body, multiple girls, tiny subject, "
    "distant view, cropped head, cropped feet, cut off, extreme perspective, complex hand pose, "
    "holding object, text, watermark, signature, logo, jpeg artifacts, photo, realistic, "
    "3d, chibi, child, mature woman, nsfw, nude, overly sexualized, giant breasts, "
    "asymmetrical eyes, crossed eyes, red eyes, blue eyes, heterochromia, deformed clothing, tangled hair"
)

# LoRA production (832x1216 -> 1664x2432 single-graph) prompts
DEFAULT_LORA_PRODUCTION_POSITIVE_PROMPT: str = (
    r"1girl, solo, rezero emilia, emilia \(re:zero\), re:zero kara hajimeru isekai seikatsu, "
    "anime key visual, official character design, silver-white hair, long hair, straight bangs, "
    "crown braid, same violet eye color, violet eyes, matching violet irises, pointed elf ears, "
    "white flower hair ornament, four-petal x-shaped hair clip, purple ribbon, "
    "white and lavender dress, white cape, detached sleeves, purple trim, green gemstone brooch, "
    "gentle gentle smile, looking at viewer, full body, medium-long shot, subject fills the frame, "
    "standing, both arms visible, one hand gently raised holding a snowflake, "
    "other hand resting near chest, visible hands, relaxed simple pose, balanced proportions, "
    "centered composition, snowy flower garden, stone balustrade, soft winter daylight, "
    "pale blue sky, clean anime lineart, refined cel shading, soft colors, masterpiece, best quality"
)

DEFAULT_LORA_PRODUCTION_NEGATIVE_PROMPT: str = (
    "worst quality, low quality, lowres, blurry, out of focus, bad anatomy, bad hands, "
    "malformed hands, fused fingers, extra fingers, missing fingers, extra arms, extra legs, "
    "extra limbs, disconnected limbs, duplicated body, multiple girls, tiny subject, "
    "distant view, cropped head, cropped feet, cut off, complex hand gesture, "
    "holding multiple objects, text, watermark, signature, logo, jpeg artifacts, photo, "
    "realistic, 3d, chibi, child, mature woman, nsfw, nude, overly sexualized, "
    "asymmetrical eyes, crossed eyes, red eyes, blue eyes, pink eyes, mismatched eyes, "
    "heterochromia, one blue eye, one pink eye, deformed clothing, tangled hair"
)

DEFAULT_LORA_PRODUCTION_FACE_POSITIVE_PROMPT: str = (
    "anime face, Emilia Re:Zero, symmetrical violet anime eyes, clear round irises, "
    "natural eye highlights, straight bangs, crown braid, white flower hair ornament, "
    "x-shaped hair clip, pointed elf ears, gentle warm smile, clean delicate lineart, "
    "refined anime cel shading, preserve silver-white hair and identity"
)

DEFAULT_LORA_PRODUCTION_HAND_POSITIVE_PROMPT: str = (
    "anime hand, natural open palm, five correctly separated fingers, clean finger joints, "
    "correct wrist, delicate pale hand, holding one snowflake, preserve sleeve and pose, "
    "clean anime lineart"
)

# Short aliases for convenience
DEFAULT_BASE_POSITIVE_PROMPT: str = DEFAULT_HIGHRES_POSITIVE_PROMPT
DEFAULT_BASE_NEGATIVE_PROMPT: str = DEFAULT_HIGHRES_NEGATIVE_PROMPT
DEFAULT_FACE_POSITIVE_PROMPT: str = DEFAULT_FACE_DETAILER_POSITIVE_PROMPT
DEFAULT_FACE_NEGATIVE_PROMPT: str = DEFAULT_FACE_DETAILER_NEGATIVE_PROMPT
DEFAULT_HAND_POSITIVE_PROMPT: str = DEFAULT_HAND_DETAILER_POSITIVE_PROMPT
DEFAULT_HAND_NEGATIVE_PROMPT: str = DEFAULT_HAND_DETAILER_NEGATIVE_PROMPT


def _resolve_root_dir() -> Path:
    env_root = os.environ.get("EMILIA_ROOT_DIR")
    if env_root:
        return Path(env_root).resolve()
    default_root = Path("D:/comfyui")
    if default_root.exists():
        return default_root
    return Path(__file__).resolve().parents[1]


def _resolve_conda_python(root_dir: Path) -> Path:
    env_py = os.environ.get("EMILIA_CONDA_PYTHON")
    if env_py:
        return Path(env_py)
    candidates = [
        Path("D:/conda-data/envs/comfyui/python.exe"),
        root_dir / "env" / "python.exe",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def _resolve_ollama_exe() -> Path:
    env_ollama = os.environ.get("EMILIA_OLLAMA_EXE")
    if env_ollama:
        return Path(env_ollama)
    primary = Path("D:/Ollama/ollama.exe")
    if primary.is_file():
        return primary
    which_ollama = shutil.which("ollama")
    if which_ollama:
        which_path = Path(which_ollama)
        if which_path.is_file():
            return which_path
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        local_candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        if local_candidate.is_file():
            return local_candidate
    return primary


@dataclass
class StudioConfig:
    """Unified configuration for Emilia Studio paths, endpoints, and canonical prompts."""

    root_dir: Path = field(default_factory=_resolve_root_dir)
    app_dir: Path | None = None
    output_dir: Path | None = None
    temp_dir: Path | None = None
    workflows_dir: Path | None = None
    visual_workflows_dir: Path | None = None
    conda_python: Path | None = None
    ollama_exe: Path | None = None

    comfy_host: str = field(
        default_factory=lambda: os.environ.get("EMILIA_COMFY_HOST", "127.0.0.1")
    )
    comfy_port: int = field(
        default_factory=lambda: int(os.environ.get("EMILIA_COMFY_PORT", "8188"))
    )
    quality_port: int = field(
        default_factory=lambda: int(os.environ.get("EMILIA_QUALITY_PORT", "8199"))
    )
    ollama_base: str = field(
        default_factory=lambda: os.environ.get("EMILIA_OLLAMA_BASE", "http://127.0.0.1:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.environ.get("EMILIA_OLLAMA_MODEL", "qwen3.5:9b")
    )

    # Canonical constants exposed on the config instance
    emilia_canonical: str = EMILIA_CANONICAL
    emilia_reference: str = EMILIA_REFERENCE
    emilia_aliases: tuple[str, ...] = EMILIA_ALIASES
    quality_tags: str = QUALITY_TAGS
    style_hints: dict[str, str] = field(default_factory=lambda: dict(STYLE_HINTS))

    # Default prompts exposed on the config instance
    base_positive_prompt: str = DEFAULT_HIGHRES_POSITIVE_PROMPT
    base_negative_prompt: str = DEFAULT_HIGHRES_NEGATIVE_PROMPT
    face_positive_prompt: str = DEFAULT_FACE_DETAILER_POSITIVE_PROMPT
    face_negative_prompt: str = DEFAULT_FACE_DETAILER_NEGATIVE_PROMPT
    hand_positive_prompt: str = DEFAULT_HAND_DETAILER_POSITIVE_PROMPT
    hand_negative_prompt: str = DEFAULT_HAND_DETAILER_NEGATIVE_PROMPT
    eye_repair_base_positive_prompt: str = DEFAULT_EYE_REPAIR_BASE_POSITIVE_PROMPT
    eye_repair_base_negative_prompt: str = DEFAULT_EYE_REPAIR_BASE_NEGATIVE_PROMPT
    eye_repair_positive_prompt: str = DEFAULT_EYE_REPAIR_POSITIVE_PROMPT
    eye_repair_negative_prompt: str = DEFAULT_EYE_REPAIR_NEGATIVE_PROMPT
    lora_native_positive_prompt: str = DEFAULT_LORA_NATIVE_POSITIVE_PROMPT
    lora_native_negative_prompt: str = DEFAULT_LORA_NATIVE_NEGATIVE_PROMPT
    lora_safe_positive_prompt: str = DEFAULT_LORA_SAFE_POSITIVE_PROMPT
    lora_safe_negative_prompt: str = DEFAULT_LORA_SAFE_NEGATIVE_PROMPT
    lora_production_positive_prompt: str = DEFAULT_LORA_PRODUCTION_POSITIVE_PROMPT
    lora_production_negative_prompt: str = DEFAULT_LORA_PRODUCTION_NEGATIVE_PROMPT

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        if self.app_dir is None:
            env_app = os.environ.get("EMILIA_APP_DIR")
            self.app_dir = Path(env_app) if env_app else self.root_dir / "app"
        else:
            self.app_dir = Path(self.app_dir)

        if self.output_dir is None:
            env_out = os.environ.get("EMILIA_OUTPUT_DIR")
            self.output_dir = Path(env_out) if env_out else self.app_dir / "output"
        else:
            self.output_dir = Path(self.output_dir)

        if self.temp_dir is None:
            env_temp = os.environ.get("EMILIA_TEMP_DIR")
            self.temp_dir = Path(env_temp) if env_temp else self.root_dir / "temp"
        else:
            self.temp_dir = Path(self.temp_dir)

        if self.workflows_dir is None:
            env_wf = os.environ.get("EMILIA_WORKFLOWS_DIR")
            if env_wf:
                self.workflows_dir = Path(env_wf)
            elif (self.root_dir / "workflows" / "api").is_dir():
                self.workflows_dir = self.root_dir / "workflows" / "api"
            else:
                self.workflows_dir = self.app_dir / "emilia"
        else:
            self.workflows_dir = Path(self.workflows_dir)

        if self.visual_workflows_dir is None:
            env_vwf = os.environ.get("EMILIA_VISUAL_WORKFLOWS_DIR")
            if env_vwf:
                self.visual_workflows_dir = Path(env_vwf)
            elif (self.root_dir / "workflows" / "visual").is_dir():
                self.visual_workflows_dir = self.root_dir / "workflows" / "visual"
            else:
                self.visual_workflows_dir = self.app_dir / "user" / "default" / "workflows"
        else:
            self.visual_workflows_dir = Path(self.visual_workflows_dir)
        if self.conda_python is None:
            self.conda_python = _resolve_conda_python(self.root_dir)
        else:
            self.conda_python = Path(self.conda_python)

        if self.ollama_exe is None:
            self.ollama_exe = _resolve_ollama_exe()
        else:
            self.ollama_exe = Path(self.ollama_exe)

        self.comfy_port = int(self.comfy_port)
        self.quality_port = int(self.quality_port)

    @property
    def comfy_base(self) -> str:
        """Base HTTP URL for standard ComfyUI service."""
        return f"http://{self.comfy_host}:{self.comfy_port}"

    @property
    def quality_base(self) -> str:
        """Base HTTP URL for isolated quality renderer ComfyUI service."""
        return f"http://{self.comfy_host}:{self.quality_port}"

    @property
    def ledger_db_path(self) -> Path:
        """Default SQLite ledger path."""
        if (self.root_dir / "workflows").is_dir():
            return self.root_dir / "workflows" / "studio_ledger.sqlite3"
        assert self.workflows_dir is not None
        return self.workflows_dir / "studio_ledger.sqlite3"

def _flatten_toml(data: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[sub_key] = sub_value
        else:
            flat[key] = value
    return flat


def load_config(config_path: str | Path | None = None) -> StudioConfig:
    """Load StudioConfig with optional TOML file overrides and environment variable precedence."""
    root = _resolve_root_dir()
    toml_overrides: dict[str, Any] = {}

    if config_path is None:
        env_cfg = os.environ.get("EMILIA_CONFIG_TOML")
        candidate = Path(env_cfg) if env_cfg else root / "config.toml"
        if candidate.is_file():
            config_path = candidate

    if config_path is not None:
        cfg_file = Path(config_path)
        if cfg_file.is_file():
            with cfg_file.open("rb") as f:
                raw_toml = tomllib.load(f)
            toml_overrides = _flatten_toml(raw_toml)

    kwargs: dict[str, Any] = {}
    valid_fields = {f.name for f in StudioConfig.__dataclass_fields__.values()}
    path_fields = {
        "root_dir",
        "app_dir",
        "output_dir",
        "temp_dir",
        "workflows_dir",
        "visual_workflows_dir",
        "conda_python",
        "ollama_exe",
    }
    env_map = {
        "root_dir": "EMILIA_ROOT_DIR",
        "app_dir": "EMILIA_APP_DIR",
        "output_dir": "EMILIA_OUTPUT_DIR",
        "temp_dir": "EMILIA_TEMP_DIR",
        "workflows_dir": "EMILIA_WORKFLOWS_DIR",
        "visual_workflows_dir": "EMILIA_VISUAL_WORKFLOWS_DIR",
        "conda_python": "EMILIA_CONDA_PYTHON",
        "ollama_exe": "EMILIA_OLLAMA_EXE",
        "comfy_host": "EMILIA_COMFY_HOST",
        "comfy_port": "EMILIA_COMFY_PORT",
        "quality_port": "EMILIA_QUALITY_PORT",
        "ollama_base": "EMILIA_OLLAMA_BASE",
        "ollama_model": "EMILIA_OLLAMA_MODEL",
    }

    for k, v in toml_overrides.items():
        if k in valid_fields:
            # Environment variables take precedence over config.toml
            env_key = env_map.get(k)
            if env_key and env_key in os.environ:
                continue
            kwargs[k] = Path(v) if k in path_fields else v

    return StudioConfig(**kwargs)
