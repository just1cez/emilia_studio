# AGENTS.md — Emilia Studio Engineering & Agent Guidelines

## 1. Repository Topology (Option A: Standalone Top-Level Studio + ComfyUI Runtime)

This repository uses an **Option A** two-tier layout separating our declarative studio orchestration (`emilia_studio/`) from the underlying ComfyUI execution engine (`app/`):

- **`emilia_studio/`** (Top-level Python package — Primary Ownership):
  - `config.py`: Path resolution, environment variable overrides (`EMILIA_*`), `config.toml` loader, and canonical Emilia character/prompt constants.
  - `workflows.py`: Declarative `EmiliaWorkflowBuilder` generating both API execution graphs (`highres_native`, `highres_finish`, `eye_repair`, `lora_native`) and synchronized GUI visual workflows (`sync_visual_workflow`).
  - `inspector.py`: Zero-VRAM (CPU-enforced) `QualityInspector` using YOLOv8n (`face_yolov8n.pt`, `hand_yolov8n.pt`) and HSV circular hue analysis to detect multi-subject hallucinations and blue/pink heterochromia eye drift.
  - `compiler.py`: `OllamaPromptCompiler` translating Chinese natural-language ideas into Danbooru-style SDXL tags with dynamic composition framing (`portrait` / `upper_body` / `full_body`) and IPAdapter weight routing.
  - `engine.py`: Two-stage isolated rendering engine (`EmiliaRenderEngine`) managing process lifecycle, WebSocket execution, quality gate routing, and `.run.json` receipt generation.
  - `ledger.py`: SQLite experiment and render ledger (`ExperimentLedger`) indexing all historical receipts and 10-seed architecture benchmark rounds.
  - `cli.py`: Unified CLI entrypoint (`python -m emilia_studio.cli` / `emilia-studio`).
- **Top-Level Tracked Assets (`workflows/`, `patches/`, `assets/`, `tests/`)**:
  - `workflows/api/`: 27 canonical `.api.json` workflow definitions.
  - `workflows/visual/`: 10 synchronized ComfyUI visual workflows (`Emilia_01..10`).
  - `workflows/reports/` & `workflows/WORKFLOW_DECISIONS.txt`: Historical benchmark reports and decision records.
  - `patches/`: `emilia_runtime_patches/` plugin plus `comfyui_core.patch` and `ipadapter_plus.patch` git diffs.
  - `assets/official_face_center_clean.png`: Canonical IPAdapter face reference image.
  - `tests/`: `test_emilia_studio.py` and `tests/comfy_core/` unit tests.
- **Root Launchers & Config**:
  - `render-emilia-quality.ps1`: PowerShell wrapper for `render_emilia_quality.py` (`-Seed`, `-Pag`, `-AutoGate`).
  - `start-comfyui.ps1`: Interactive GUI server launcher on port `8188`.
  - `ollama_comfyui_bridge.py`: CLI bridge driving either the 2-stage HighRes Quality pipeline (`--mode quality`, default) or single-stage preview (`--mode fast`).
  - `config.example.toml` & `models_manifest.json`: Portable configuration template and model weight/checksum manifest.
- **`app/`** (Git-Ignored External ComfyUI Runtime Subtree):
  - Standalone upstream ComfyUI checkout (`Comfy-Org/ComfyUI`, retaining its own `app/.git`) plus pinned custom nodes (`comfyui-ipadapter-plus`, `comfyui-impact-pack`, `comfyui-impact-subpack`).
  - Run `python -m emilia_studio.cli setup --sync` to sync top-level `assets/`, `patches/`, and `workflows/visual/` into `app/`.
  - When modifying anything inside `app/comfy/` or `app/nodes.py`, also follow `app/AGENTS.md` and regenerate `patches/comfyui_core.patch`.
---

## 2. Hardware, Memory, and Process Isolation Rules

Target workstation baseline: **8 GiB NVIDIA CMP 40HX (Turing, compute capability 7.5)** + **Intel Xeon E5-2690 v2 (10C/20T)** on Windows with bounded system commit limits.

1. **Mandatory Server Flags**:
   - Always launch ComfyUI with:
     `--reserve-vram 0.8 --fp32-vae --disable-pinned-memory --disable-api-nodes`
   - Automated batch/quality renders MUST pass `--preview-method none` (GUI launcher `start-comfyui.ps1` retains `--preview-method latent2rgb`).
   - Do **not** enable pinned memory or static VRAM modes; historical trials exhausted the Windows commit limit regardless of free physical RAM.
2. **Two-Stage Process Isolation**:
   - **Stage 1 (`native`)** loads Illustrious-XL + Emilia LoRA + CLIP-ViT-H + IPAdapter Plus Face (`1152×1728`, 36 steps, DPM++ 2M SDE Karras, tiled FP32 VAE decode `tile_size=512, overlap=64`).
   - **Stage 2 (`finish`)** loads Illustrious-XL + Emilia LoRA + YOLOv8n (`face`/`hand`) + SAM (`sam_vit_b_01ec64.pth`) + `FaceDetailer` + `DetailerForEach` + `RealESRGAN_x4plus_anime_6B.pth` (4× upscale followed by 0.5× Lanczos downsample to `2304×3456`).
   - Never keep Stage 1 and Stage 2 models co-resident in the same process under the 8 GiB VRAM / Windows commit budget. Stage 1 child process must exit cleanly before Stage 2 starts.
3. **Zero-VRAM Pre-Flight & Inspection**:
   - **Ollama**: Must always be invoked with `"options": {"num_gpu": 0, ...}` and `"keep_alive": 0` so LLM weights never touch GPU VRAM.
   - **QualityInspector**: Any YOLO face/hand inference inside `emilia_studio/inspector.py` MUST explicitly pass `device="cpu"` (`use_yolo=True` on CPU only).
4. **No Unsupported Kernel Experiments**:
   - Do not install xFormers, Triton, SageAttention, or FP8/INT8 quantization kernels for this Turing GPU. PyTorch native memory-efficient SDPA is verified pixel-identical and optimal on CC 7.5.
   - Do not re-enable `UltimateSDUpscale` or `TiledDiffusion` in `app/custom_nodes/`; their audited disabled copies reside in `temp/community_quality/*-tested-disabled/` so global sampler hooks stay clean.

---

## 3. Workflow & Quality Invariants

1. **Single Source of Truth for Workflows**:
   - Do not proliferate ad-hoc static JSON copies for minor seed, weight, or resolution tweaks. Use `EmiliaWorkflowBuilder` in `emilia_studio/workflows.py`.
   - Default calls to `build_highres_native(seed=20261027, pag=False)`, `build_highres_finish(image_path=..., seed=20261027, pag=False)`, and `build_eye_repair(...)` **must remain 100% key-and-value identical** to the canonical `.api.json` definitions in `app/emilia/`.
   - Both API and GUI workflows must explicitly set `noise_mask_feather: 0` on `FaceDetailer` (node `21`) and `DetailerForEach` (node `28`), while keeping paste-back `feather: 12`. Use `builder.sync_visual_workflow()` when updating GUI JSONs in `app/user/default/workflows/`.
2. **Bit-Exact Pixel Reproducibility**:
   - For fixed seed `20261027` (`pag=False`, `strict_seed_lock=True`), any performance or architectural change must preserve exact decoded PNG SHA-256 hashes (`hashlib.sha256(Image.open(p).tobytes()).hexdigest()`):
     - Native (`1152×1728`): `3b58d9daf2770d0ad7e87b971912b854e1e7495f9e99a09f348af9bf258c9425`
     - Final (`2304×3456`): `d7fe10c200535d8656ab633248ca7cc651ee6d2e60bd309c23c43fa1ce5af63e`
3. **Quality Gate Operating Rule**:
   - Never upscale malformed anatomy in unattended creative mode (`--auto-gate`).
   - `QualityInspector` flags `multiple_subjects` (`face_count > 1` on solo prompts, e.g., seed `20261030`) or missing faces as `reject_seed`, and flags heterochromia (`heterochromia_score > 32.0°`) or non-violet irises (`hue` outside `[250°, 302°]`) as `repair_eyes` (routing through `build_eye_repair` at `denoise=0.42`).
   - In benchmark / default fixed-seed mode (`strict_seed_lock=True`), `QualityInspector` records full diagnostics into the `.run.json` receipt and SQLite ledger without altering execution flow or pixel hashes.

---

## 4. Verification & Testing Requirements

Before completing any code or workflow change, run the unit test suite using the project Conda environment:

```powershell
# 1. Run focused Emilia Studio + runtime optimization unit tests (~20s)
D:\conda-data\envs\comfyui\python.exe -m pytest app/tests-unit/comfy_test/test_emilia_studio.py app/tests-unit/comfy_test/test_input_content_hash.py app/tests-unit/comfy_test/test_save_image_quantization.py app/tests-unit/comfy_test/test_tiled_scale_blending.py app/tests-unit/comfy_test/test_vae_output_ownership.py

# 2. Run the full 147-test unit suite in app/tests-unit/comfy_test/
D:\conda-data\envs\comfyui\python.exe -m pytest app/tests-unit/comfy_test/

# 3. Verify CLI health (workflows, inspector, and SQLite ledger)
D:\conda-data\envs\comfyui\python.exe -m emilia_studio.cli workflows
D:\conda-data\envs\comfyui\python.exe -m emilia_studio.cli ledger stats
```

## 5. Git & Repository Hygiene

- Never commit model weights (`*.safetensors`, `*.pth`, `*.pt`, `*.task`), local Conda environments (`env/`), runtime caches (`cache/`, `temp/`), SQLite databases (`*.sqlite3`, `*.db`), or generated image outputs (`app/output/`).
- Keep `config.toml` local-only (ignored by Git); commit schema changes to `config.example.toml`.
- Document every workflow, quality, or performance finding in `PROGRESS.md` and `app/emilia/WORKFLOW_DECISIONS.txt` with exact seed numbers, wall times, and decoded pixel SHA-256 hashes.
