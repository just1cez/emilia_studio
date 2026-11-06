"""Workflow construction and auditing for Emilia Studio."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .config import StudioConfig, load_config

@dataclass
class WorkflowSpec:
    """Declarative specification for Emilia workflows."""
    graph: Dict[str, Any]
    visual_template: Optional[Path] = None

_RAW_WORKFLOW_CACHE: dict[str, str] = {}


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
                        if key in inputs:
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
