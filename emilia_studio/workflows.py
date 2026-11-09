"""Workflow construction and auditing for Emilia Studio."""

from __future__ import annotations

import json
from copy import deepcopy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .config import EMILIA_ANGLE_SPECS, ULTRA_FRAMING_LEADS, StudioConfig, load_config

@dataclass
class WorkflowSpec:
    """Declarative specification for Emilia workflows."""
    graph: Dict[str, Any]
    visual_template: Optional[Path] = None

_RAW_WORKFLOW_CACHE: dict[str, str] = {}


def _choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}; got {value!r}")


def _range(name: str, value: float, lower: float, upper: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite and between {lower} and {upper}")


def _model_name(name: str, value: str) -> None:
    # ComfyUI resolves category-relative names, including configured extra model paths.
    # Keep an explicit selection intact: a missing model must fail, never substitute.
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty model filename")
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or ":" in normalized or any(part in ("", ".", "..") for part in normalized.split("/")):
        raise ValueError(f"{name} must be a category-relative model filename")


class EmiliaWorkflowBuilder:
    """Builder for constructing standard Emilia API graphs."""
    
    def __init__(self, config: StudioConfig | None = None):
        self.config = config or load_config()

    def _load_api(self, name: str) -> Dict[str, Any]:
        path = self.config.workflows_dir / name
        key = str(path.resolve())
        raw = _RAW_WORKFLOW_CACHE.get(key)
        if raw is None:
            raw = path.read_text(encoding="utf-8")
            _RAW_WORKFLOW_CACHE[key] = raw
        return json.loads(raw)
    def build_highres_native(
        self,
        seed: int = 20261027,
        pag: bool = False,
        pag_scale: float = 1.0,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        width: int = 1152,
        height: int = 1728,
        steps: int = 36,
        cfg: float = 5.5,
        ipadapter_weight: float = 0.4,
        ipadapter_end_at: float = 0.75,
        filename_prefix: str | None = None
    ) -> Dict[str, Any]:
        graph = self._load_api("emilia_ipadapter_highres_production.api.json")
        graph["3"]["inputs"]["seed"] = seed
        graph["3"]["inputs"]["steps"] = steps
        graph["3"]["inputs"]["cfg"] = cfg
        graph["5"]["inputs"].update(width=width, height=height)
        graph["42"]["inputs"].update(weight=ipadapter_weight, end_at=ipadapter_end_at)
        if positive_prompt:
            graph["6"]["inputs"]["text"] = positive_prompt
        if negative_prompt:
            graph["7"]["inputs"]["text"] = negative_prompt
        if filename_prefix:
            graph["9"]["inputs"]["filename_prefix"] = filename_prefix
        
        if pag:
            graph["60"] = {
                "class_type": "PerturbedAttentionGuidance",
                "inputs": {"model": graph["3"]["inputs"]["model"], "scale": pag_scale}
            }
            graph["3"]["inputs"]["model"] = ["60", 0]
            if not filename_prefix:
                graph["9"]["inputs"]["filename_prefix"] = "emilia/community/quality_pag_native"
        
        return graph

    def build_highres_finish(
        self,
        image_path: str,
        seed: int = 20261027,
        pag: bool = False,
        face_denoise: float = 0.22,
        hand_denoise: float = 0.24,
        face_steps: int = 20,
        hand_steps: int = 20,
        upscale_factor: float = 2.0
    ) -> Dict[str, Any]:
        graph = self._load_api("emilia_ipadapter_highres_upscale.api.json")
        graph["1"]["inputs"]["image"] = image_path
        graph["21"]["inputs"]["seed"] = seed + 3
        graph["21"]["inputs"]["denoise"] = face_denoise
        graph["21"]["inputs"]["steps"] = face_steps
        graph["28"]["inputs"]["seed"] = seed + 4
        graph["28"]["inputs"]["denoise"] = hand_denoise
        graph["28"]["inputs"]["steps"] = hand_steps
        if upscale_factor != 2.0 and "15" in graph and "inputs" in graph["15"]:
            graph["15"]["inputs"]["scale_by"] = round(float(upscale_factor) / 4.0, 4)
        if pag:
            graph["29"]["inputs"]["filename_prefix"] = "emilia/community/quality_pag_refined"
            graph["16"]["inputs"]["filename_prefix"] = "emilia/community/quality_pag_finished"
            
        return graph

    def build_eye_repair(
        self,
        image_path: str,
        seed: int = 20260936,
        denoise: float = 0.42,
        steps: int = 16,
        guide_size: int = 640,
        upscale: bool = True
    ) -> Dict[str, Any]:
        graph = self._load_api("emilia_eye_repair.api.json")
        graph["1"]["inputs"]["image"] = image_path
        graph["21"]["inputs"].update({
            "seed": seed,
            "denoise": denoise,
            "steps": steps,
            "guide_size": guide_size
        })
        if not upscale:
            for node_id in ("13", "14", "15", "16"):
                graph.pop(node_id, None)
        return graph

    def build_lora_native(
        self,
        positive_prompt: str,
        seed: int,
        width: int = 832,
        height: int = 1216,
        steps: int = 24,
        cfg: float = 5.5,
        filename_prefix: str = "ollama_animagine"
    ) -> Dict[str, Any]:
        graph = self._load_api("emilia_lora_native.api.json")
        graph["3"]["inputs"].update(seed=seed, steps=steps, cfg=cfg)
        graph["5"]["inputs"].update(width=width, height=height)
        graph["6"]["inputs"]["text"] = positive_prompt
        graph["9"]["inputs"]["filename_prefix"] = filename_prefix
        return graph


    def build_ultra_native(
        self,
        seed: int = 20261027,
        pag: bool = False,
        pag_scale: float = 0.8,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        base_width: int = 832,
        base_height: int = 1216,
        width: int = 1152,
        height: int = 1728,
        base_steps: int = 28,
        hires_steps: int = 18,
        cfg_base: float = 4.5,
        cfg_hires: float = 4.5,
        hires_denoise: float = 0.40,
        rescale_cfg_multiplier: float = 0.75,
        lora_name: str | None = None,
        lora_strength_base: float = 0.25,
        lora_strength_hires: float = 0.0,
        ipadapter_weight: float = 0.18,
        skirt_ipadapter_weight: float = 0.45,
        ipadapter_start_at: float = 0.15,
        ipadapter_end_at: float = 0.60,
        filename_prefix: str | None = None,
        framing: str = "full_body",
        camera_angle: str = "auto",
        reference_mode: str = "single",
        hires_bridge: str = "lanczos",
        checkpoint_name: str = "Illustrious-XL-v0.1.safetensors",
        downscale_factor: float | None = None,
    ) -> Dict[str, Any]:
        """Build ultra native; experimental references/bridge/downscale are opt-in.

        Only front and three-quarter face references are verified. Other angles
        affect text conditioning, not a promise of pose-matched face imagery.
        Portrait omits outfit IPA; upper-body moves its soft mask to the torso.
        ``downscale_factor`` enables Kohya Deep Shrink on the base pass only.
        """
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("seed must be a non-negative 64-bit integer")
        _choice("framing", framing, ("full_body", "upper_body", "portrait"))
        _choice("camera_angle", camera_angle, ("auto", "front", "three-quarter", "side", "high-angle", "low-angle"))
        _choice("reference_mode", reference_mode, ("single", "paired"))
        _choice("hires_bridge", hires_bridge, ("lanczos", "anime"))
        _model_name("checkpoint_name", checkpoint_name)
        if lora_name is not None:
            _model_name("lora_name", lora_name)
        if downscale_factor is not None:
            _range("downscale_factor", downscale_factor, 1.0, 9.0)
            if downscale_factor == 1.0:
                raise ValueError("downscale_factor must be greater than 1, or None to disable")
        for name, value in (("base_width", base_width), ("base_height", base_height), ("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or not 64 <= value <= 16384 or value % 8:
                raise ValueError(f"{name} must be an integer multiple of 8 between 64 and 16384")
        for name, value in (("base_steps", base_steps), ("hires_steps", hires_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10000:
                raise ValueError(f"{name} must be an integer between 1 and 10000")
        for name, value in (("hires_denoise", hires_denoise), ("rescale_cfg_multiplier", rescale_cfg_multiplier), ("ipadapter_start_at", ipadapter_start_at), ("ipadapter_end_at", ipadapter_end_at)):
            _range(name, value, 0.0, 1.0)
        if ipadapter_start_at > ipadapter_end_at:
            raise ValueError("ipadapter_start_at must not exceed ipadapter_end_at")
        for name, value in (("ipadapter_weight", ipadapter_weight), ("skirt_ipadapter_weight", skirt_ipadapter_weight)):
            _range(name, value, 0.0, 3.0)
        if skirt_ipadapter_weight > 0 and framing != "portrait" and ipadapter_end_at < 0.10:
            raise ValueError("ipadapter_end_at must be at least 0.10 for outfit conditioning")
        for name, value in (("cfg_base", cfg_base), ("cfg_hires", cfg_hires), ("pag_scale", pag_scale)):
            _range(name, value, 0.0, 100.0)
        for name, value in (("lora_strength_base", lora_strength_base), ("lora_strength_hires", lora_strength_hires)):
            _range(name, value, -100.0, 100.0)
        pos = positive_prompt or self.config.ultra_positive_prompt
        neg = negative_prompt or self.config.ultra_negative_prompt
        if not positive_prompt and framing != "full_body":
            pos = pos.replace(ULTRA_FRAMING_LEADS["full_body"], ULTRA_FRAMING_LEADS[framing])
            if framing == "portrait":
                pos = ", ".join(
                    t.strip()
                    for t in pos.split(",")
                    if not any(w in t for w in ("skirt", "stockings", "boots", "thighs", "zettai", "coattails", "legs"))
                )
        if not negative_prompt and framing != "full_body":
            neg = ", ".join(
                t.strip()
                for t in neg.split(",")
                if t.strip() not in ("cropped feet", "feet out of frame")
            )
        if camera_angle == "auto":
            lowered = pos.casefold()
            for resolved, cues in (("high-angle", ("from above", "high-angle")), ("low-angle", ("from below", "low-angle")), ("side", ("from side", "side profile", "looking back")), ("three-quarter", ("three-quarter", "three quarter", "3/4"))):
                if any(cue in lowered for cue in cues):
                    camera_angle = resolved
                    break
        angle_spec = EMILIA_ANGLE_SPECS[camera_angle]
        if camera_angle != "auto" and angle_spec["tags"]:
            existing_tags = {t.strip().casefold() for t in pos.split(",") if t.strip()}
            missing_angle_tags = [
                t.strip()
                for t in angle_spec["tags"].split(",")
                if t.strip() and t.strip().casefold() not in existing_tags
            ]
            if missing_angle_tags:
                pos = f"{pos}, {', '.join(missing_angle_tags)}"
        verified_faces = ("official_face_center_clean.png", "emilia_official_face_3q.png")
        primary_ref = angle_spec["primary_ref"]
        if primary_ref not in verified_faces:
            raise ValueError(f"No verified face reference configured for {camera_angle!r}")
        # Never feed the registry's action/full-body or unverified angle images to face IPA.
        secondary_ref = next(ref for ref in verified_faces if ref != primary_ref)
        out_prefix = filename_prefix or (
            "emilia/ultra/01_pag_base" if pag else "emilia/ultra/01_base"
        )
        app_dir = self.config.app_dir
        if lora_name is None:
            if app_dir is not None and (app_dir / "models" / "loras" / "Emilia_Rezero_Q-000005.safetensors").is_file():
                lora_name = "Emilia_Rezero_Q-000005.safetensors"
            else:
                lora_name = "emilia-re-zero-re.safetensors"

        has_general_ipa = bool(
            app_dir is not None
            and (app_dir / "models" / "ipadapter" / "ip-adapter-plus_sdxl_vit-h.safetensors").is_file()
        )
        has_skirt_ref = bool(
            (app_dir is not None and (app_dir / "input" / "emilia_official_skirt_and_bodice.png").is_file())
            or (self.config.assets_dir / "official_references" / "emilia_official_skirt_and_bodice.png").is_file()
        )
        use_skirt_ipa = framing != "portrait" and skirt_ipadapter_weight > 0 and has_general_ipa and has_skirt_ref

        graph: Dict[str, Any] = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": primary_ref},
                "_meta": {"title": "Official clean center face reference" if primary_ref == "official_face_center_clean.png" else "Verified three-quarter face identity reference (not exact camera-pose guidance)"},
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint_name},
                "_meta": {"title": "Illustrious XL v0.1" if checkpoint_name == "Illustrious-XL-v0.1.safetensors" else checkpoint_name},
            },
            "10": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["4", 0],
                    "clip": ["4", 1],
                    "lora_name": lora_name,
                    "strength_model": lora_strength_base,
                    "strength_clip": lora_strength_base,
                },
                "_meta": {"title": "Emilia Canonical Outfit LoRA (Base Pass)"},
            },
            "11": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["4", 0],
                    "clip": ["4", 1],
                    "lora_name": lora_name,
                    "strength_model": lora_strength_hires,
                    "strength_clip": lora_strength_hires,
                },
                "_meta": {"title": "Emilia LoRA (1152x1728 Hires Refinement Pass)"},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": pos, "clip": ["10", 1]},
                "_meta": {"title": "Pass 1: Canonical outfit + scene conditioning"},
            },
            "61": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": pos,
                    "clip": ["4", 1] if lora_strength_hires == 0 else (["10", 1] if lora_strength_hires == lora_strength_base else ["11", 1]),
                },
                "_meta": {"title": "Pass 2: Painterly Illustrious lighting & detail conditioning"},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": neg, "clip": ["4", 1]},
                "_meta": {"title": "Anatomy, footwear, cloak & overexposure guardrails"},
            },
            "40": {
                "class_type": "IPAdapterModelLoader",
                "inputs": {"ipadapter_file": "ip-adapter-plus-face_sdxl_vit-h.safetensors"},
                "_meta": {"title": "IPAdapter Plus Face SDXL"},
            },
            "41": {
                "class_type": "CLIPVisionLoader",
                "inputs": {"clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"},
                "_meta": {"title": "CLIP Vision ViT-H"},
            },
            "42": {
                "class_type": "IPAdapterAdvanced",
                "inputs": {
                    "model": ["10", 0],
                    "ipadapter": ["40", 0],
                    "image": ["1", 0],
                    "weight": ipadapter_weight,
                    "weight_type": "ease in-out",
                    "combine_embeds": "average",
                    "start_at": ipadapter_start_at,
                    "end_at": ipadapter_end_at,
                    "embeds_scaling": "V only",
                    "clip_vision": ["41", 0],
                },
                "_meta": {"title": "Gentle V-only ease in-out face anchor"},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": base_width, "height": base_height, "batch_size": 1},
                "_meta": {"title": "SDXL Native 832x1216 Proportion Canvas"},
            },
        }
        if reference_mode == "paired" and ipadapter_weight > 0:
            graph["100"] = {
                "class_type": "LoadImage",
                "inputs": {"image": secondary_ref},
                "_meta": {"title": "Complementary verified front/three-quarter face"},
            }
            for node_id, source in (("101", "1"), ("102", "100")):
                graph[node_id] = {
                    "class_type": "PrepImageForClipVision",
                    "inputs": {"image": [source, 0], "interpolation": "LANCZOS", "crop_position": "center", "sharpening": 0.0},
                }
            graph["104"] = {
                "class_type": "ImageBatch",
                "inputs": {"image1": ["101", 0], "image2": ["102", 0]},
            }
            graph["42"]["inputs"]["image"] = ["104", 0]
            graph["42"]["_meta"]["title"] = "Average verified face embeddings (one output image)"
        base_model_ref: list[Any] = ["42", 0] if ipadapter_weight > 0 else ["10", 0]

        if use_skirt_ipa:
            graph["2"] = {
                "class_type": "LoadImage",
                "inputs": {"image": "emilia_official_skirt_and_bodice.png"},
                "_meta": {"title": "Official Emilia bodice, 3-tier scalloped peplum & pleated skirt sheet"},
            }
            graph["103"] = {
                "class_type": "PrepImageForClipVision",
                "inputs": {
                    "image": ["2", 0],
                    "interpolation": "LANCZOS",
                    "crop_position": "center",
                    "sharpening": 0.1,
                },
                "_meta": {"title": "Prepare official skirt reference for CLIP Vision"},
            }
            graph["200"] = {
                "class_type": "SolidMask",
                "inputs": {"value": 0.0, "width": base_width, "height": base_height},
            }
            graph["201"] = {
                "class_type": "SolidMask",
                "inputs": {
                    "value": 1.0,
                    "width": round(base_width * (0.60 if framing == "upper_body" else 0.42)),
                    "height": round(base_height * (0.44 if framing == "upper_body" else 0.34)),
                },
            }
            graph["202"] = {
                "class_type": "MaskComposite",
                "inputs": {
                    "destination": ["200", 0],
                    "source": ["203", 0],
                    "x": round(base_width * (0.20 if framing == "upper_body" else 0.29)),
                    "y": round(base_height * (0.45 if framing == "upper_body" else 0.25)),
                    "operation": "add",
                },
            }
            graph["203"] = {
                "class_type": "FeatherMask",
                "inputs": {
                    "mask": ["201", 0],
                    "left": min(96, graph["201"]["inputs"]["width"] // 2),
                    "top": min(96, graph["201"]["inputs"]["height"] // 2),
                    "right": min(96, graph["201"]["inputs"]["width"] // 2),
                    "bottom": min(96, graph["201"]["inputs"]["height"] // 2),
                },
                "_meta": {"title": "Spatial torso/skirt attention mask (protects background lighting)"},
            }
            graph["43"] = {
                "class_type": "IPAdapterModelLoader",
                "inputs": {"ipadapter_file": "ip-adapter-plus_sdxl_vit-h.safetensors"},
                "_meta": {"title": "General SDXL IPAdapter Plus (Outfit & Skirt Geometry)"},
            }
            graph["44"] = {
                "class_type": "IPAdapterAdvanced",
                "inputs": {
                    "model": base_model_ref,
                    "ipadapter": ["43", 0],
                    "image": ["103", 0],
                    "weight": skirt_ipadapter_weight,
                    "weight_type": "ease in-out",
                    "combine_embeds": "norm average",
                    "start_at": 0.10,
                    "end_at": ipadapter_end_at,
                    "embeds_scaling": "V only",
                    "attn_mask": ["202", 0],
                    "clip_vision": ["41", 0],
                },
                "_meta": {"title": "Spatial-masked official skirt & bodice conditioning"},
            }
            base_model_ref = ["44", 0]

        if pag:
            graph["60"] = {
                "class_type": "PerturbedAttentionGuidance",
                "inputs": {"model": base_model_ref, "scale": pag_scale},
            }
            base_model_ref = ["60", 0]

        if rescale_cfg_multiplier > 0:
            graph["46"] = {
                "class_type": "RescaleCFG",
                "inputs": {"model": base_model_ref, "multiplier": rescale_cfg_multiplier},
                "_meta": {"title": "RescaleCFG (prevents highlight blowout & neon saturation)"},
            }
            base_model_ref = ["46", 0]

        if downscale_factor is not None:
            graph["47"] = {
                "class_type": "PatchModelAddDownscale",
                "inputs": {
                    "model": base_model_ref, "block_number": 3,
                    "downscale_factor": downscale_factor, "start_percent": 0.0,
                    "end_percent": 0.35, "downscale_after_skip": True,
                    "downscale_method": "bicubic", "upscale_method": "bicubic",
                },
                "_meta": {"title": "Experimental base-only Kohya Deep Shrink"},
            }
            base_model_ref = ["47", 0]

        graph["3"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": base_model_ref,
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": base_steps,
                "cfg": cfg_base,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "denoise": 1.0,
            },
            "_meta": {"title": "Pass 1: 832x1216 Global Anatomy & Official Skirt Render"},
        }
        graph["50"] = {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
            "_meta": {"title": "FP32 Tiled Decode 832x1216"},
        }
        graph["51"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["50", 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "disabled",
            },
            "_meta": {"title": "Lanczos Pixel Upscale to 1152x1728"},
        }
        if hires_bridge == "anime":
            graph["56"] = {
                "class_type": "UpscaleModelLoader",
                "inputs": {"model_name": "RealESRGAN_x4plus_anime_6B.pth"},
            }
            graph["57"] = {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {"upscale_model": ["56", 0], "image": ["50", 0]},
            }
            graph["51"]["inputs"]["image"] = ["57", 0]
            graph["51"]["_meta"]["title"] = "Experimental Anime bridge, Lanczos resized to exact hires canvas"
        graph["52"] = {
            "class_type": "VAEEncodeTiled",
            "inputs": {
                "pixels": ["51", 0],
                "vae": ["4", 2],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
            "_meta": {"title": "FP32 Tiled Encode 1152x1728"},
        }
        hires_model_ref: list[Any] = (
            ["4", 0]
            if lora_strength_hires == 0
            else (["10", 0] if lora_strength_hires == lora_strength_base else ["11", 0])
        )
        if rescale_cfg_multiplier > 0:
            graph["55"] = {
                "class_type": "RescaleCFG",
                "inputs": {"model": hires_model_ref, "multiplier": rescale_cfg_multiplier},
                "_meta": {"title": "Pass 2 RescaleCFG for luminous painterly lighting"},
            }
            hires_model_ref = ["55", 0]

        graph["53"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": hires_model_ref,
                "positive": ["61", 0],
                "negative": ["7", 0],
                "latent_image": ["52", 0],
                "seed": seed + 1,
                "steps": hires_steps,
                "cfg": cfg_hires,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": hires_denoise,
            },
            "_meta": {"title": "Pass 2: Full-Canvas 1152x1728 Painterly Lighting & Skirt Hires.fix"},
        }
        graph["8"] = {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["53", 0],
                "vae": ["4", 2],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
            "_meta": {"title": "FP32 Tiled Decode 1152x1728"},
        }
        graph["9"] = {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": out_prefix,
            },
            "_meta": {"title": "Save 1152x1728 Multi-Scale Base"},
        }
        return graph

    def build_ultra_finish(
        self,
        image_path: str,
        seed: int = 20261027,
        pag: bool = False,
        lora_name: str | None = None,
        lora_strength: float = 0.25,
        face_denoise: float = 0.28,
        hand_denoise: float = 0.30,
        face_steps: int = 20,
        hand_steps: int = 20,
        face_guide_size: int = 768,
        face_max_size: int = 1024,
        hand_guide_size: int = 640,
        hand_max_size: int = 896,
        upscale_factor: float = 2.0,
        checkpoint_name: str = "Illustrious-XL-v0.1.safetensors",
        garment_region: str | None = None,
        garment_denoise: float = 0.18,
        garment_steps: int = 20,
        upscale_blend_strength: float = 0.0,
        upscaler_name: str = "RealESRGAN_x4plus_anime_6B.pth",
        secondary_upscaler_name: str | None = None,
    ) -> Dict[str, Any]:
        """Build ultra finishing without adding any IPA or SAM model loads.

        ``garment_region`` opts into a face-relative geometry prior, not garment
        detection. It reuses the face mask and hand detailer's sampling settings.
        ``upscale_blend_strength`` mixes Lanczos (or ``secondary_upscaler_name``)
        into the primary upscaler result. Zero keeps the existing one-model path
        and never loads the secondary model. Both branches use the exact target
        size, independent of the selected models' native scale factors.
        Explicit model selections are passed through without fallback.
        """
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("seed must be a non-negative 64-bit integer")
        _model_name("checkpoint_name", checkpoint_name)
        _model_name("upscaler_name", upscaler_name)
        if secondary_upscaler_name is not None:
            _model_name("secondary_upscaler_name", secondary_upscaler_name)
        if lora_name is not None:
            _model_name("lora_name", lora_name)
        if garment_region is not None:
            _choice("garment_region", garment_region, ("brooch_and_collar", "bodice_and_peplum", "full_garment"))
        for name, value in (("garment_denoise", garment_denoise), ("face_denoise", face_denoise), ("hand_denoise", hand_denoise), ("upscale_blend_strength", upscale_blend_strength)):
            _range(name, value, 0.0, 1.0)
        _range("lora_strength", lora_strength, -100.0, 100.0)
        _range("upscale_factor", upscale_factor, 0.04, 8.0)
        for name, value in (("face_steps", face_steps), ("hand_steps", hand_steps), ("garment_steps", garment_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10000:
                raise ValueError(f"{name} must be an integer between 1 and 10000")
        for name, value in (("face_guide_size", face_guide_size), ("face_max_size", face_max_size), ("hand_guide_size", hand_guide_size), ("hand_max_size", hand_max_size)):
            if isinstance(value, bool) or not isinstance(value, int) or not 64 <= value <= 16384:
                raise ValueError(f"{name} must be an integer between 64 and 16384")
        if face_guide_size > face_max_size or hand_guide_size > hand_max_size:
            raise ValueError("Detailer guide sizes must not exceed their max sizes")
        app_dir = self.config.app_dir
        if lora_name is None:
            if app_dir is not None and (app_dir / "models" / "loras" / "Emilia_Rezero_Q-000005.safetensors").is_file():
                lora_name = "Emilia_Rezero_Q-000005.safetensors"
            else:
                lora_name = "emilia-re-zero-re.safetensors"
        graph = self.build_highres_finish(
            image_path=image_path,
            seed=seed,
            pag=pag,
            face_denoise=face_denoise,
            hand_denoise=hand_denoise,
            face_steps=face_steps,
            hand_steps=hand_steps,
            upscale_factor=upscale_factor,
        )
        graph["4"]["inputs"]["ckpt_name"] = checkpoint_name
        if checkpoint_name != "Illustrious-XL-v0.1.safetensors":
            graph["4"]["_meta"]["title"] = checkpoint_name
        graph["13"]["inputs"]["model_name"] = upscaler_name
        if upscaler_name != "RealESRGAN_x4plus_anime_6B.pth":
            graph["13"]["_meta"]["title"] = upscaler_name
        graph["10"]["inputs"]["lora_name"] = lora_name
        graph["10"]["inputs"]["strength_model"] = lora_strength
        graph["10"]["inputs"]["strength_clip"] = lora_strength
        graph["17"]["inputs"]["text"] = self.config.ultra_face_positive_prompt
        graph["18"]["inputs"]["text"] = self.config.ultra_face_negative_prompt
        graph["21"]["inputs"]["guide_size"] = face_guide_size
        graph["21"]["inputs"]["max_size"] = face_max_size
        graph["21"]["inputs"]["cfg"] = 5.0
        graph["26"]["inputs"]["text"] = self.config.ultra_hand_positive_prompt
        graph["27"]["inputs"]["text"] = self.config.ultra_hand_negative_prompt
        graph["28"]["inputs"]["guide_size"] = hand_guide_size
        graph["28"]["inputs"]["max_size"] = hand_max_size
        graph["28"]["inputs"]["cfg"] = 5.0
        refined_image: list[Any] = ["28", 0]
        if garment_region is not None:
            graph["70"] = {
                "class_type": "EmiliaRelativeGarmentSEGS",
                "inputs": {
                    "image": refined_image, "face_mask": ["21", 3],
                    "region": garment_region, "crop_factor": 1.6, "feather_px": 16,
                },
                "_meta": {"title": "Experimental face-relative garment geometry prior (not detection)"},
            }
            graph["71"] = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": self.config.ultra_brooch_positive_prompt if garment_region == "brooch_and_collar" else f"{self.config.ultra_outfit_canonical}, preserve original garment layout, preserve fabric colors and folds, delicate clean anime lineart",
                    "clip": deepcopy(graph["26"]["inputs"]["clip"]),
                },
            }
            graph["72"] = {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": self.config.ultra_brooch_negative_prompt if garment_region == "brooch_and_collar" else self.config.ultra_negative_prompt, "clip": deepcopy(graph["27"]["inputs"]["clip"])},
            }
            garment_inputs = deepcopy(graph["28"]["inputs"])
            garment_inputs.update(
                image=refined_image, segs=["70", 0], positive=["71", 0],
                negative=["72", 0], seed=seed + 5,
                denoise=garment_denoise, steps=garment_steps,
            )
            graph["73"] = {
                "class_type": "DetailerForEach",
                "inputs": garment_inputs,
                "_meta": {"title": "Experimental local garment refinement"},
            }
            refined_image = ["73", 0]
            graph["14"]["inputs"]["image"] = refined_image
            graph["29"]["inputs"]["images"] = refined_image
        if upscaler_name != "RealESRGAN_x4plus_anime_6B.pth":
            # A selected model need not be 4x. Derive the final canvas from the
            # refined image instead of assuming its model's enlargement factor.
            graph["85"] = {
                "class_type": "ImageScaleBy",
                "inputs": {"image": refined_image, "upscale_method": "lanczos", "scale_by": upscale_factor},
            }
            graph["86"] = {"class_type": "GetImageSize", "inputs": {"image": ["85", 0]}}
            graph["15"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["14", 0], "upscale_method": "lanczos",
                    "width": ["86", 0], "height": ["86", 1], "crop": "disabled",
                },
            }
        if upscale_blend_strength > 0:
            # Read actual primary output dimensions: multiplying two scale factors
            # independently can round differently on arbitrary source dimensions.
            graph["80"] = {"class_type": "GetImageSize", "inputs": {"image": ["15", 0]}}
            blend_source = refined_image
            if secondary_upscaler_name is not None:
                graph["83"] = {
                    "class_type": "UpscaleModelLoader",
                    "inputs": {"model_name": secondary_upscaler_name},
                }
                graph["84"] = {
                    "class_type": "ImageUpscaleWithModel",
                    "inputs": {"upscale_model": ["83", 0], "image": refined_image},
                }
                blend_source = ["84", 0]
            graph["81"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": blend_source, "upscale_method": "lanczos",
                    "width": ["80", 0], "height": ["80", 1], "crop": "disabled",
                },
            }
            graph["82"] = {
                "class_type": "ImageBlend",
                "inputs": {
                    "image1": ["15", 0], "image2": ["81", 0],
                    "blend_factor": upscale_blend_strength, "blend_mode": "normal",
                },
                "_meta": {"title": "Experimental dimension-matched upscaler blend"},
            }
            graph["16"]["inputs"]["images"] = ["82", 0]
        if not pag:
            graph["29"]["inputs"]["filename_prefix"] = "emilia/ultra/02_refined"
            graph["16"]["inputs"]["filename_prefix"] = "emilia/ultra/03_finished_2x"
        return graph
    def sync_visual_workflow(self, api_graph: Dict[str, Any], visual_template_path: Path) -> Dict[str, Any]:
        """Synchronizes API graph values into a visual template."""
        visual = json.loads(visual_template_path.read_text(encoding="utf-8"))
        for node in visual.get("nodes", []):
            node_id = str(node["id"])
            if node_id in api_graph:
                inputs = api_graph[node_id].get("inputs", {})
                wvn = node.get("widgets_values_named")
                if wvn is not None:
                    for key in wvn:
                        if key in inputs and not isinstance(inputs[key], list):
                            wvn[key] = inputs[key]
                    node["widgets_values"] = list(wvn.values())
        return visual

    def audit_workflow_directory(self) -> Dict[str, list[str]]:
        """Classifies all JSON files in workflows/api and workflows/reports (or app/emilia)."""
        results: Dict[str, list[str]] = {
            "canonical_workflows": [],
            "historical_workflows": [],
            "experiment_reports": [],
        }
        search_dirs = [self.config.workflows_dir]
        reports_dir = self.config.root_dir / "workflows" / "reports"
        if reports_dir.is_dir() and reports_dir not in search_dirs:
            search_dirs.append(reports_dir)
        seen: set[str] = set()
        for d in search_dirs:
            if d is None or not d.is_dir():
                continue
            for p in sorted(d.glob("*.json")):
                name = p.name
                if name in seen:
                    continue
                seen.add(name)
                if name.endswith(".api.json"):
                    results["canonical_workflows"].append(name)
                elif any(sub in name for sub in ["report", "test", "experiment", "evaluation", "run"]):
                    results["experiment_reports"].append(name)
                else:
                    results["historical_workflows"].append(name)
        return results
