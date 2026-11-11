"""Ollama Chinese-to-SDXL structured prompt compiler for Emilia Studio."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_HIGHRES_NEGATIVE_PROMPT,
    DEFAULT_ULTRA_NEGATIVE_PROMPT,
    EMILIA_ALIASES,
    EMILIA_CANONICAL,
    EMILIA_OUTFIT_MODES,
    EMILIA_COMPANION_TAGS,
    EMILIA_MAGIC_TAGS,
    EMILIA_ANGLE_SPECS,
    QUALITY_TAGS,
    STYLE_HINTS,
    ULTRA_EMILIA_OUTFIT_CANONICAL,
    ULTRA_FRAMING_LEADS,
    StudioConfig,
    load_config,
)

_IDEA_TAG_MEM_CACHE: dict[str, str] = {}

class BridgeError(RuntimeError):
    """An expected bridge or local-service failure."""


@dataclass
class CompiledPromptSpec:
    """Structured prompt specification compiled from a natural-language idea."""

    user_idea: str
    is_emilia: bool
    style: str
    scene_and_pose_tags: str
    positive_prompt: str
    negative_prompt: str
    framing: str = "full_body"
    recommended_width: int = 1152
    recommended_height: int = 1728
    recommended_ipadapter_weight: float = 0.40
    recommended_pag: bool = False
    model_used: str = "qwen3.5:9b"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def http_json(
    url: str,
    *,
    method: str = "GET",
    value: Any = None,
    timeout: float = 30,
) -> Any:
    data = None if value is None else json.dumps(value, ensure_ascii=False).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BridgeError(f"HTTP {exc.code} from {url}: {detail[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise BridgeError(f"无法连接 {url}: {exc.reason}") from exc
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeError(f"{url} 返回的不是有效 JSON") from exc


def is_emilia_request(idea: str, aliases: tuple[str, ...] = EMILIA_ALIASES) -> bool:
    lowered = idea.casefold()
    return any(alias.casefold() in lowered for alias in aliases)


def clean_generated_prompt(content: str) -> str:
    prompt = content.strip()
    lowered = prompt.lower()
    while "<think>" in lowered and "</think>" in lowered:
        start = lowered.index("<think>")
        end_idx = lowered.find("</think>", start)
        if end_idx == -1:
            break
        end = end_idx + len("</think>")
        prompt = (prompt[:start] + prompt[end:]).strip()
        lowered = prompt.lower()
    if "</think>" in lowered:
        end = lowered.rindex("</think>") + len("</think>")
        prompt = prompt[end:].strip()
        lowered = prompt.lower()
    if "<think>" in lowered:
        start = lowered.index("<think>")
        prompt = prompt[:start].strip()
        lowered = prompt.lower()
    if prompt.startswith("```"):
        prompt = prompt.strip("`").strip()
        lowered = prompt.lower()
        for lang in ("json\n", "text\n", "markdown\n"):
            if lowered.startswith(lang):
                prompt = prompt[len(lang) :].strip()
                lowered = prompt.lower()
    for prefix in ("positive prompt:", "prompt:", "英文提示词:"):
        if lowered.startswith(prefix):
            prompt = prompt[len(prefix) :].strip()
            lowered = prompt.lower()
    for marker in ("\nnegative prompt:", "\n负面提示词:"):
        if marker in lowered:
            prompt = prompt[: lowered.index(marker)].strip()
            lowered = prompt.lower()
    prompt = " ".join(line.strip() for line in prompt.splitlines() if line.strip())
    if not prompt:
        raise BridgeError("Ollama 返回了空的生图提示词")
    if len(prompt) > 4000:
        prompt = prompt[:4000].rsplit(" ", 1)[0]
    return prompt


def deduplicate_tags(prompt: str) -> str:
    parts = [part.strip() for part in prompt.split(",") if part.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part.casefold()
        if key not in seen:
            seen.add(key)
            deduped.append(part)
    return ", ".join(deduped)


class OllamaPromptCompiler:
    """Translates Chinese or natural-language ideas into SDXL / Illustrious-XL prompt specs."""

    def __init__(self, config: StudioConfig | None = None) -> None:
        self.config = config or load_config()

    def probe_ollama(self) -> dict[str, Any] | None:
        try:
            tags = http_json(f"{self.config.ollama_base}/api/tags", timeout=10)
        except BridgeError as exc:
            if "无法连接" in str(exc):
                return None
            raise
        if not isinstance(tags, dict) or "models" not in tags:
            raise BridgeError("11434 端口已有服务，但它不是可识别的 Ollama")
        return tags

    def ensure_ollama(self) -> dict[str, Any]:
        tags = self.probe_ollama()
        if tags is None:
            executable = self.config.ollama_exe
            if executable is None or not executable.is_file():
                raise BridgeError(f"Ollama 未运行，且未找到 {executable}，无法自动启动")
            print("Ollama 未运行，正在启动本地服务……", flush=True)
            assert self.config.temp_dir is not None
            self.config.temp_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.config.temp_dir / "ollama_bridge_server.log"
            child_env = os.environ.copy()
            child_env.update(
                {
                    "TEMP": str(self.config.temp_dir),
                    "TMP": str(self.config.temp_dir),
                }
            )
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            with log_path.open("a", encoding="utf-8") as log:
                subprocess.Popen(
                    [str(executable), "serve"],
                    cwd=str(executable.parent),
                    env=child_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    creationflags=creation_flags,
                    close_fds=True,
                )
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                time.sleep(0.25)
                tags = self.probe_ollama()
                if tags is not None:
                    break
            if tags is None:
                raise BridgeError(f"Ollama 启动超时；请查看 {log_path}")
        models = {
            item.get("name")
            for item in tags.get("models", [])
            if isinstance(item, dict)
        }
        if self.config.ollama_model not in models:
            raise BridgeError(
                f"Ollama 未找到模型 {self.config.ollama_model}，当前模型列表：{sorted(models)!r}"
            )
        return tags

    def infer_composition_hints(
        self, idea: str, tags: str, *, ultra_mode: bool = False, outfit: str = "royal"
    ) -> tuple[str, float]:
        """Infer framing (portrait/upper_body/full_body) and optimal IPAdapter face weight."""
        combined = f"{idea} {tags}".casefold()
        outfit_ipa = float(EMILIA_OUTFIT_MODES.get(outfit, {}).get("ipadapter_weight", 0.18))
        if any(k in combined for k in ("特写", "头像", "面部", "close-up", "portrait", "face focus")):
            return "portrait", (max(0.20, outfit_ipa) if ultra_mode else 0.50)
        if any(k in combined for k in ("半身", "腰部以上", "upper body", "cowboy shot", "medium shot")):
            return "upper_body", (outfit_ipa if ultra_mode else 0.45)
        return "full_body", (outfit_ipa if ultra_mode else 0.40)

    def build_emilia_ultra_prompt(
        self, scene_and_pose_tags: str, framing: str = "full_body", *,
        outfit: str = "royal", companion: str = "none", magic: str = "none", angle: str = "auto",
    ) -> str:
        """Compose Emilia identity and explicit creative controls without contradictory framing."""
        if framing not in ULTRA_FRAMING_LEADS:
            raise ValueError(f"Unknown framing: {framing}")
        for value, registry in ((outfit, EMILIA_OUTFIT_MODES), (companion, EMILIA_COMPANION_TAGS), (magic, EMILIA_MAGIC_TAGS), (angle, EMILIA_ANGLE_SPECS)):
            if value not in registry:
                raise ValueError(f"Unknown creative option: {value}")
        identity = self.config.ultra_outfit_canonical if outfit == 'royal' else EMILIA_OUTFIT_MODES[outfit]['positive']
        if framing == 'portrait':
            identity = ', '.join(tag.strip() for tag in identity.split(',') if not any(word in tag for word in ('skirt', 'stockings', 'boots', 'thighs', 'zettai', 'coattails', 'legs')))
        framing_lead = ULTRA_FRAMING_LEADS[framing]
        if outfit == "nightgown" and framing == "full_body":
            framing_lead = "full body, entire figure visible, generous headroom, natural proportions"
        elif outfit not in ("royal", "sanctuary") and framing == "upper_body":
            framing_lead = "upper body, cowboy shot, generous headroom above head, entire head, waist, and upper dress visible in frame"
        subject = '1girl, solo' if companion == 'none' else '1girl, one human girl with one small cat spirit companion'
        return deduplicate_tags(
            f"masterpiece, best quality, absurdres, {subject}, {framing_lead}, "
            f"{identity}, {scene_and_pose_tags}, {EMILIA_ANGLE_SPECS[angle]['tags']}, "
            f"{EMILIA_COMPANION_TAGS[companion]}, {EMILIA_MAGIC_TAGS[magic]}, "
            "balanced body proportions, coherent anatomy, delicate clean lineart, layered anime cel shading, "
            "balanced exposure, coherent light direction, natural shadow gradients"
        )

    def build_emilia_ultra_negative(self, *, outfit: str = 'royal', framing: str = 'full_body') -> str:
        if outfit not in EMILIA_OUTFIT_MODES or framing not in ULTRA_FRAMING_LEADS:
            raise ValueError('Unknown outfit or framing')
        spec = EMILIA_OUTFIT_MODES[outfit]
        negative = self.config.ultra_negative_prompt
        remove_set = {
            t.strip().casefold()
            for chunk in spec['negative_remove']
            for t in chunk.split(',')
            if t.strip()
        }
        if framing != 'full_body':
            remove_set.update({'cropped feet', 'feet out of frame'})
        filtered_neg = ', '.join(
            tag.strip()
            for tag in negative.split(',')
            if tag.strip() and tag.strip().casefold() not in remove_set
        )
        return deduplicate_tags(f"{filtered_neg}, {spec['negative_add']}")

    def compile_idea(
        self,
        idea: str,
        style: str = "anime",
        highres_quality_mode: bool = True,
        ultra_mode: bool = True,
        *,
        outfit: str = "royal",
        companion: str = "none",
        magic: str = "none",
        angle: str = "auto",
        use_cache: bool = True,
    ) -> CompiledPromptSpec:
        """Translate a Chinese idea into a full CompiledPromptSpec and unload Ollama from VRAM."""
        style_hint = self.config.style_hints.get(style, STYLE_HINTS["anime"])
        cache_key = hashlib.sha256(
            f"v1|{self.config.ollama_model}|{style}|{idea.strip()}".encode("utf-8")
        ).hexdigest()
        cache_file = (
            (self.config.temp_dir / "ollama_prompt_cache.json")
            if self.config.temp_dir is not None
            else None
        )
        raw_tags: str | None = _IDEA_TAG_MEM_CACHE.get(cache_key) if use_cache else None
        if raw_tags is None and use_cache and cache_file is not None and cache_file.is_file():
            try:
                disk_cache = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(disk_cache, dict) and isinstance(disk_cache.get(cache_key), str):
                    raw_tags = disk_cache[cache_key]
                    _IDEA_TAG_MEM_CACHE[cache_key] = raw_tags
            except (OSError, ValueError):
                pass

        if raw_tags is None:
            logical_cpus = os.cpu_count() or 8
            phys_threads = min(10, max(2, logical_cpus // 2))
            body = {
                "model": self.config.ollama_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Translate the Chinese image idea into concise English Danbooru-style "
                            "tags for the local Illustrious XL model with the Emilia Re:Zero LoRA. "
                            "Output only one comma-separated line without explanations or markdown. "
                            "Preserve the requested pose, framing, scene and lighting. Use no more "
                            "than 45 tags. Do not include unsafe sexual or graphic violent content. "
                            "For Emilia, output scene and pose tags only: canonical character tags "
                            "are added separately. Do not redesign her hair or clothes. "
                            "For other subjects start with subject count, character and series. "
                            "Do not add quality tags; they are appended separately. "
                            f"Rendering direction: {style_hint}."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Translate this idea only:\n<user_idea>{idea}</user_idea>",
                    },
                ],
                "stream": False,
                "think": False,
                # Keep the prompt model on CPU and unload it before SDXL starts so the
                # 8 GB graphics card is reserved for image generation.
                "options": {
                    "temperature": 0.2,
                    "num_gpu": 0,
                    "num_thread": phys_threads,
                    "num_ctx": 2048,
                    "num_predict": 192,
                },
                "keep_alive": 0,
            }
            response = http_json(
                f"{self.config.ollama_base}/api/chat",
                method="POST",
                value=body,
                timeout=900,
            )
            try:
                content = response["message"]["content"]
            except (KeyError, TypeError) as exc:
                raise BridgeError(f"Ollama 返回格式不完整：{response!r}") from exc
            if not isinstance(content, str):
                raise BridgeError("Ollama 返回的提示词不是文本")

            raw_tags = clean_generated_prompt(content)
            if use_cache:
                _IDEA_TAG_MEM_CACHE[cache_key] = raw_tags
                if cache_file is not None:
                    try:
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        existing_disk: dict[str, str] = {}
                        if cache_file.is_file():
                            loaded = json.loads(cache_file.read_text(encoding="utf-8"))
                            if isinstance(loaded, dict):
                                existing_disk = {str(k): str(v) for k, v in loaded.items() if isinstance(v, str)}
                        existing_disk[cache_key] = raw_tags
                        cache_file.write_text(
                            json.dumps(existing_disk, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except OSError:
                        pass
        emilia_flag = is_emilia_request(idea, self.config.emilia_aliases)
        framing, ipa_weight = self.infer_composition_hints(
            idea, raw_tags, ultra_mode=(highres_quality_mode and ultra_mode), outfit=outfit
        )

        if emilia_flag:
            if highres_quality_mode and ultra_mode:
                combined = self.build_emilia_ultra_prompt(
                    raw_tags,
                    framing=framing,
                    outfit=outfit,
                    companion=companion,
                    magic=magic,
                    angle=angle,
                )
                neg_prompt = self.build_emilia_ultra_negative(outfit=outfit, framing=framing)
            elif highres_quality_mode:
                highres_identity = (
                    r"masterpiece, best quality, anime illustration, emilia \(re:zero\), 1girl, solo, "
                    "silver-white long hair, crown braid, straight bangs, matching violet eyes, "
                    "pointed elf ears, white flower hair ornament, x-shaped hair clip, purple ribbons, "
                    "white and lavender dress with purple trim, green gemstone brooch, detached bell sleeves"
                )
                combined = deduplicate_tags(f"{highres_identity}, {raw_tags}, {self.config.quality_tags}")
                neg_prompt = self.config.base_negative_prompt
            else:
                combined = deduplicate_tags(f"{self.config.emilia_canonical}, {raw_tags}")
                combined = f"{combined}, {self.config.quality_tags}"
                neg_prompt = DEFAULT_HIGHRES_NEGATIVE_PROMPT
        else:
            combined = f"{deduplicate_tags(raw_tags)}, {self.config.quality_tags}"
            base_neg = DEFAULT_ULTRA_NEGATIVE_PROMPT if ultra_mode else DEFAULT_HIGHRES_NEGATIVE_PROMPT
            emilia_specific_bans = {
                "hood", "hooded cloak", "white cloak", "fur hood", "cape hood",
                "purple bodysuit", "bodysuit", "solid purple corset", "dark purple apron",
                "floor-length gown", "wedding dress", "emblem on skirt", "emblem on coattails",
                "four purple tails", "round gold button", "double brooch",
                "red eyes", "blue eyes", "solid blue eyes", "pink eyes", "heterochromia",
            }
            neg_prompt = ", ".join(
                t.strip()
                for t in base_neg.split(",")
                if t.strip() and t.strip().casefold() not in emilia_specific_bans
            )

        return CompiledPromptSpec(
            user_idea=idea,
            is_emilia=emilia_flag,
            style=style,
            scene_and_pose_tags=raw_tags,
            positive_prompt=combined,
            negative_prompt=neg_prompt,
            framing=framing,
            recommended_width=1152 if highres_quality_mode else 832,
            recommended_height=1728 if highres_quality_mode else 1216,
            recommended_ipadapter_weight=ipa_weight,
            recommended_pag=False,
            model_used=self.config.ollama_model,
        )
