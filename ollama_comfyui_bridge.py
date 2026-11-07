#!/usr/bin/env python3
"""Translate a Chinese idea with Ollama and render via Emilia Studio (HighRes Quality 2x or Fast LoRA)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

from emilia_studio.compiler import (
    BridgeError,
    OllamaPromptCompiler,
    clean_generated_prompt,
    http_json,
    is_emilia_request,
)
from emilia_studio.config import (
    EMILIA_ALIASES,
    EMILIA_CANONICAL,
    EMILIA_REFERENCE,
    QUALITY_TAGS,
    STYLE_HINTS,
    load_config,
)
from emilia_studio.engine import EmiliaRenderEngine
from emilia_studio.workflows import EmiliaWorkflowBuilder

CONFIG = load_config()
ROOT = CONFIG.root_dir
APP = CONFIG.app_dir
ENV = CONFIG.conda_python.parent if CONFIG.conda_python else ROOT / "env"
PYTHON = CONFIG.conda_python or (ROOT / "env" / "python.exe")
OUTPUT = CONFIG.output_dir
TEMP = CONFIG.temp_dir
COMFY_BASE = CONFIG.comfy_base
OLLAMA_BASE = CONFIG.ollama_base
OLLAMA_MODEL = CONFIG.ollama_model
NATIVE_WORKFLOW = CONFIG.workflows_dir / "emilia_lora_native.api.json"
VISUAL_WORKFLOW = CONFIG.visual_workflows_dir / "Emilia_03_LoRA_8GB.json"


def probe_comfy() -> dict[str, Any] | None:
    """Return system stats when ComfyUI is up, or None only on connection refusal."""
    try:
        stats = http_json(f"{COMFY_BASE}/system_stats", timeout=10)
    except BridgeError as exc:
        if "无法连接" in str(exc):
            return None
        raise
    if not isinstance(stats, dict) or "system" not in stats:
        raise BridgeError("8188 端口已有服务，但它不是可识别的 ComfyUI")
    return stats


def probe_ollama() -> dict[str, Any] | None:
    """Return the Ollama tag response when up, or None only on connection refusal."""
    return OllamaPromptCompiler(CONFIG).probe_ollama()


def start_background(command: list[str], cwd: Path, log_path: Path) -> None:
    assert TEMP is not None
    TEMP.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env.update(
        {
            "PIP_CACHE_DIR": str(ROOT / "cache" / "pip"),
            "TEMP": str(TEMP),
            "TMP": str(TEMP),
        }
    )
    log = log_path.open("a", encoding="utf-8")
    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        subprocess.Popen(
            command,
            cwd=str(cwd),
            env=child_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creation_flags,
            close_fds=True,
        )
    finally:
        log.close()


def ensure_comfy() -> dict[str, Any]:
    assert APP is not None and TEMP is not None
    stats = probe_comfy()
    if stats is None:
        if not PYTHON.is_file() or not (APP / "main.py").is_file():
            raise BridgeError(f"找不到 ComfyUI 环境或入口：{PYTHON} / {APP / 'main.py'}")
        print("ComfyUI 未运行，正在启动一个本地实例……", flush=True)
        start_background(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ROOT / "start-comfyui.ps1"),
            ],
            APP,
            TEMP / "comfyui_bridge_server.log",
        )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            time.sleep(2)
            stats = probe_comfy()
            if stats is not None:
                break
        if stats is None:
            raise BridgeError(
                f"ComfyUI 启动超时；请查看 {TEMP / 'comfyui_bridge_server.log'}"
            )
    models = http_json(f"{COMFY_BASE}/models/checkpoints", timeout=20)
    checkpoint = json.loads(NATIVE_WORKFLOW.read_text(encoding="utf-8"))["4"]["inputs"]["ckpt_name"]
    if checkpoint not in models:
        raise BridgeError(
            f"ComfyUI 未找到模型 {checkpoint}；模型仍在下载或需要重启ComfyUI"
        )
    return stats


def ensure_ollama() -> dict[str, Any]:
    return OllamaPromptCompiler(CONFIG).ensure_ollama()


def generate_sdxl_prompt(
    idea: str,
    style: str = "anime",
    *,
    highres_quality_mode: bool = False,
) -> str:
    compiler = OllamaPromptCompiler(CONFIG)
    spec = compiler.compile_idea(
        idea,
        style=style,
        highres_quality_mode=highres_quality_mode,
    )
    return spec.positive_prompt


def queue_image(
    idea: str,
    positive_prompt: str,
    *,
    width: int,
    height: int,
    steps: int,
    output_prefix: str = "ollama_animagine",
    seed: int | None = None,
) -> str:
    if seed is None:
        seed = secrets.randbelow(2**53)
    builder = EmiliaWorkflowBuilder(CONFIG)
    graph = builder.build_lora_native(
        positive_prompt=positive_prompt,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        filename_prefix=output_prefix,
    )
    visual = builder.sync_visual_workflow(graph, VISUAL_WORKFLOW)
    for node in visual.get("nodes", []):
        widgets = node.get("widgets_values_named", {})
        if "control_after_generate" in widgets:
            widgets["control_after_generate"] = "fixed"
            node["widgets_values"] = list(widgets.values())

    body = {
        "prompt": graph,
        "extra_data": {
            "extra_pnginfo": {
                "workflow": visual,
                "generation": {
                    "bridge": "ollama_comfyui_bridge",
                    "reference": EMILIA_REFERENCE if is_emilia_request(idea) else None,
                    "ollama_model": OLLAMA_MODEL,
                    "user_idea": idea,
                },
            }
        },
    }
    queued = http_json(
        f"{COMFY_BASE}/prompt",
        method="POST",
        value=body,
        timeout=30,
    )
    if queued.get("node_errors"):
        raise BridgeError(
            "ComfyUI 节点错误：" + json.dumps(queued["node_errors"], ensure_ascii=False)
        )
    prompt_id = queued.get("prompt_id")
    if not isinstance(prompt_id, str):
        raise BridgeError(f"ComfyUI 没有返回 prompt_id：{queued!r}")
    print(f"已提交 ComfyUI，随机种子：{seed}", flush=True)
    return prompt_id


def wait_for_image(prompt_id: str, *, timeout: float = 3600) -> list[Path]:
    assert OUTPUT is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = http_json(
            f"{COMFY_BASE}/history/{urllib.parse.quote(prompt_id, safe='')}",
            timeout=30,
        )
        if prompt_id not in history:
            time.sleep(2)
            continue
        record = history[prompt_id]
        status = record.get("status", {})
        if status.get("status_str") != "success" or not status.get("completed"):
            raise BridgeError(
                "ComfyUI 生图失败：" + json.dumps(status, ensure_ascii=False)
            )
        descriptors = record.get("outputs", {}).get("9", {}).get("images", [])
        if not descriptors:
            raise BridgeError("ComfyUI 报告成功，但 SaveImage 没有输出图片")
        output_root = OUTPUT.resolve()
        paths: list[Path] = []
        for descriptor in descriptors:
            filename = descriptor.get("filename")
            subfolder = descriptor.get("subfolder", "")
            if not isinstance(filename, str) or not isinstance(subfolder, str):
                raise BridgeError(f"ComfyUI 返回了无效图片描述：{descriptor!r}")
            path = (OUTPUT / subfolder / filename).resolve()
            if output_root not in path.parents:
                raise BridgeError("拒绝使用越出 output 目录的图片路径")
            if not path.is_file():
                raise BridgeError(f"ComfyUI 返回的图片文件不存在：{path}")
            paths.append(path)
        return paths
    raise BridgeError("等待 ComfyUI 生图超过 60 分钟")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 Ollama 把中文想法改写成提示词，并自动提交 ComfyUI 生图（支持 HighRes Quality 2x 精修管线与快速单阶段模式）。"
    )
    parser.add_argument("idea", nargs="+", help="中文生图想法")
    parser.add_argument(
        "--mode",
        choices=("ultra", "quality", "fast"),
        default="ultra",
        help="ultra: 832->1152 多尺度Hires+服饰锁定+面手精修+2304x3456超分 (默认); quality: 旧版1152直出; fast: 832x1216单阶段",
    )
    parser.add_argument("--seed", type=int, default=None, help="指定随机种子")
    parser.add_argument("--pag", action="store_true", help="在 quality 模式下启用 PAG scale 1.0")
    parser.add_argument("--auto-gate", action="store_true", help="启用自动瞳色修复与结构质检门禁")
    parser.add_argument("--width", type=int, choices=range(64, 4097, 8), help="默认使用已验证工作流宽度")
    parser.add_argument("--height", type=int, choices=range(64, 4097, 8), help="默认使用已验证工作流高度")
    parser.add_argument("--steps", type=int, choices=range(1, 101), help="默认使用已验证工作流步数")
    parser.add_argument(
        "--style",
        choices=("anime", "illustration", "cinematic"),
        default="anime",
        help="生成风格提示，默认为日系动画原画风格",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    assert APP is not None
    args = parse_args(argv)
    idea = " ".join(args.idea).strip()
    if not idea:
        raise BridgeError("请输入中文生图想法")
    if not PYTHON.is_file() or not (APP / "main.py").is_file():
        raise BridgeError("D:\\comfyui 的 ComfyUI 安装不完整")

    print("检查 Ollama 本地服务……", flush=True)
    compiler = OllamaPromptCompiler(CONFIG)
    compiler.ensure_ollama()

    if args.mode in ("ultra", "quality"):
        is_ultra = args.mode == "ultra"
        print(f"正在让 Ollama 编译高画质结构化提示词 [{args.mode}]……", flush=True)
        spec = compiler.compile_idea(
            idea,
            style=args.style,
            highres_quality_mode=True,
            ultra_mode=is_ultra,
        )
        print(
            f"英文提示词 [{args.mode}/{spec.framing}, IPAdapter={spec.recommended_ipadapter_weight}]：{spec.positive_prompt}",
            flush=True,
        )
        seed = args.seed if args.seed is not None else secrets.randbelow(2**31 - 1)
        engine = EmiliaRenderEngine(CONFIG)
        receipt = asyncio.run(
            engine.render_quality_pipeline(
                seed=seed,
                port=CONFIG.quality_port,
                pag=args.pag,
                profile=args.mode,
                positive_prompt=spec.positive_prompt,
                negative_prompt=spec.negative_prompt,
                ipadapter_weight=spec.recommended_ipadapter_weight,
                user_idea=idea,
                strict_seed_lock=not args.auto_gate,
                auto_eye_repair=args.auto_gate,
                max_seed_retries=2 if args.auto_gate else 0,
            )
        )
        print(f"母版图片：{receipt['native']}", flush=True)
        print(f"终版 2304×3456 图片：{receipt['final']}", flush=True)
        return 0

    # Fast single-stage mode
    defaults = json.loads(NATIVE_WORKFLOW.read_text(encoding="utf-8"))
    width = args.width or defaults["5"]["inputs"]["width"]
    height = args.height or defaults["5"]["inputs"]["height"]
    steps = args.steps or defaults["3"]["inputs"]["steps"]
    print("检查 ComfyUI 本地服务……", flush=True)
    ensure_comfy()
    print("正在让 Ollama 生成英文提示词……", flush=True)
    positive_prompt = generate_sdxl_prompt(idea, args.style, highres_quality_mode=False)
    print(f"英文提示词：{positive_prompt}", flush=True)
    prefix = "emilia/ollama_native" if is_emilia_request(idea) else "ollama_animagine"
    prompt_id = queue_image(
        idea,
        positive_prompt,
        width=width,
        height=height,
        steps=steps,
        output_prefix=prefix,
        seed=args.seed,
    )
    print(
        f"正在生成 {width}×{height}，{steps} 步；单张运行，避免8GB显存争用……",
        flush=True,
    )
    paths = wait_for_image(prompt_id)
    for path in paths:
        print(f"图片已生成：{path}", flush=True)
    print("提示词和工作流已写入 PNG 元数据。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BridgeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
