"""Configuration and canonical constants for Emilia Studio."""

from __future__ import annotations

import json
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

# Ultra Multi-Scale Hires.fix (832x1216 -> 1152x1728 -> 2304x3456) canonical identity, official skirt geometry & lighting lock
ULTRA_EMILIA_OUTFIT_CANONICAL: str = (
    r"emilia_rezero, rezero emilia, emilia \(re:zero\), re:zero kara hajimeru isekai seikatsu, "
    "official character design, anime key visual, silver-white hair, very long hair, straight bangs, "
    "crown braid, symmetrical violet eyes with cyan-blue inner pupil ring, clear round purple irises, natural eye highlights, pointed elf ears, "
    "white rose hair ornament, x hair ornament, purple hair ribbon, "
    "detached white high collar with scalloped purple trim, green diamond crystal pendant at throat, bare shoulders, "
    "white sweetheart bustier cups with purple frilled trim, "
    "white corset waist panel with six vertical purple dash markings, "
    "three-tiered white petal-scalloped peplum overskirt with lavender edges, "
    "white box-pleated miniskirt with thin purple stripe near hem, "
    "plain white split dress coattails with thin purple border trim and pale lavender lining, "
    "detached wide white juliet bell sleeves with purple frilled upper cuffs and pointed purple sleeve cuffs bearing yellow hawk emblem, "
    "zettai ryouiki, bare thighs, white thighhigh stockings with purple zig-zag V-shaped cuffs, "
    "white high-heel ankle boots with purple V-cuffs"
)

ULTRA_FRAMING_LEADS: dict[str, str] = {
    "full_body": (
        "full body, medium-long shot, entire body visible from head to toe, "
        "generous headroom above head, both feet and white ankle boots completely visible on ground"
    ),
    "upper_body": (
        "upper body, cowboy shot, generous headroom above head, entire head, waist, and pleated skirt visible in frame"
    ),
    "portrait": (
        "portrait, upper body, generous headroom above head, entire head and hair flower fully in frame"
    ),
}

DEFAULT_ULTRA_POSITIVE_PROMPT: str = (
    "masterpiece, best quality, absurdres, polished anime key visual illustration, 1girl, solo, "
    f"{ULTRA_FRAMING_LEADS['full_body']}, {ULTRA_EMILIA_OUTFIT_CANONICAL}, "
    "standing on a sunlit stone garden path, blooming white roses, stone archway, "
    "intricate background foliage, soft golden hour volumetric light, warm key light and cool ambient shadows, "
    "natural shadow gradients on face and dress pleats, balanced exposure, delicate clean lineart, luminous shading"
)

DEFAULT_ULTRA_NEGATIVE_PROMPT: str = (
    "worst quality, low quality, lowres, unfinished, sketch, blurry, overexposed, neon greens, oversaturated, "
    "text, watermark, signature, copyright, copyright name, logo, artist name, english text, "
    "flat lighting, harsh contrast, blown-out highlights, anime screencap, flat background, simple background, "
    "white background, glowing outline, halo around body, "
    "bad anatomy, bad proportions, long torso, long legs, warped legs, bowed legs, knock-kneed, "
    "extra legs, third leg, extra limbs, fused legs, asymmetrical boots, mismatched footwear, "
    "stocking feet, shoeless, barefoot, cropped feet, feet out of frame, "
    "bad hands, malformed hands, fused fingers, extra fingers, missing fingers, "
    "hood, hooded cloak, white cloak, fur hood, cape hood, purple bodysuit, bodysuit, "
    "solid purple corset, dark purple apron, floor-length gown, wedding dress, "
    "emblem on skirt, emblem on coattails, four purple tails, "
    "round gold button, double brooch, background character, distant person, crowd, multiple girls, "
    "asymmetrical eyes, crossed eyes, spiral eyes, target eyes, multiple pupils, "
    "red eyes, solid blue eyes, pink eyes, heterochromia, cropped head, photo, realistic, 3d, nsfw"
)
DEFAULT_ULTRA_FACE_POSITIVE_PROMPT: str = (
    r"masterpiece, best quality, rezero emilia, emilia \(re:zero\), anime face, "
    "symmetrical violet eyes, clear glossy purple outer irises with luminous cyan-blue inner ring around soft vertical oval pupils, "
    "bright white eye highlights, delicate eyelashes, subtle nose and lip shading, "
    "natural warm smile, silver-white hair, straight bangs, crown braid, "
    "white flower hair ornament, x hair ornament, pointed elf ears, "
    "clean delicate lineart, layered anime cel shading"
)

DEFAULT_ULTRA_FACE_NEGATIVE_PROMPT: str = (
    "worst quality, low quality, blurry face, "
    "spiral eyes, target eyes, multiple pupils, asymmetrical eyes, crossed eyes, "
    "red eyes, solid blue eyes, pink eyes, heterochromia, dot mouth, flat shading, "
    "distorted face, deformed mouth, realistic, photo, 3d, text, watermark, signature, copyright"
)

DEFAULT_ULTRA_HAND_POSITIVE_PROMPT: str = (
    "masterpiece, best quality, anime hands, five slender fingers, "
    "clear finger separation, natural knuckles and fingernails, correct wrist anatomy, "
    "pale fair skin, white and purple sleeve cuff, clean delicate lineart, layered anime cel shading"
)

DEFAULT_ULTRA_HAND_NEGATIVE_PROMPT: str = (
    "worst quality, low quality, malformed hand, extra fingers, missing fingers, "
    "fused fingers, six fingers, four fingers, mutated hands, twisted wrist, "
    "duplicated hand, extra arm, blurry, realistic, photo, text, watermark"
)

DEFAULT_ULTRA_BROOCH_POSITIVE_PROMPT: str = (
    r"masterpiece, best quality, emilia_rezero, rezero emilia, emilia \(re:zero\), "
    "detached white high collar with scalloped purple trim and small gold throat clasp, "
    "faceted glowing green emerald diamond crystal pendant suspended at center of chest, "
    "bare shoulders, white sweetheart bustier cups with purple frilled trim, "
    "white corset waist panel with six vertical purple dash markings, "
    "crisp symmetrical jewelry facets, clean delicate anime lineart, luminous cel shading"
)

DEFAULT_ULTRA_BROOCH_NEGATIVE_PROMPT: str = (
    "worst quality, low quality, blurry brooch, asymmetrical crystal, red gem, blue gem, purple gem, "
    "round gold button, double brooch, missing pendant, deformed collar, smudged lineart, realistic, photo, text, watermark"
)

# Canonical Emilia Outfit Registry (--outfit)
EMILIA_OUTFIT_MODES: dict[str, dict[str, Any]] = {
    "royal": {
        "title": "王选标准白紫礼服 (Canonical Royal Selection Dress)",
        "positive": ULTRA_EMILIA_OUTFIT_CANONICAL,
        "negative_remove": (),
        "negative_add": (
            "hood, hooded cloak, white cloak, fur hood, cape hood, purple bodysuit, bodysuit, "
            "solid purple corset, dark purple apron, floor-length gown, wedding dress"
        ),
        "use_skirt_ipa": True,
        "use_brooch_detailer": True,
        "lora_strength": 0.25,
        "lora_strength_base": 0.25,
        "lora_strength_finish": 0.25,
        "ipadapter_weight": 0.18,
        "reference_mode": "single",
    },
    "cloak": {
        "title": "第一季认识阻碍白兜帽斗篷 (S1 White Hooded Cloak)",
        "positive": (
            r"emilia_rezero, rezero emilia, emilia \(re:zero\), re:zero kara hajimeru isekai seikatsu, "
            "official character design, anime key visual, silver-white hair, very long hair, straight bangs, "
            "crown braid, symmetrical violet eyes, clear round irises, natural eye highlights, pointed elf ears, "
            "white rose hair ornament, x hair ornament, purple hair ribbon, "
            "wearing ornate white hooded cloak with purple trim and gold clasp at neck, "
            "flowing white cape over white and purple dress, green diamond crystal pendant at throat, "
            "white box-pleated miniskirt, white thighhigh stockings, white high-heel ankle boots with purple V-cuffs"
        ),
        "negative_remove": ("hood, hooded cloak, white cloak, fur hood, cape hood, ",),
        "negative_add": "modern jacket, hoodie, zipper, dark black cloak",
        "use_skirt_ipa": False,
        "use_brooch_detailer": False,
        "lora_strength": 0.15,
        "lora_strength_base": 0.15,
        "lora_strength_finish": 0.25,
        "ipadapter_weight": 0.22,
        "reference_mode": "paired",
    },
    "frozen-bond": {
        "title": "冰结森林主题创作服装（非官设复刻） / Frozen-forest interpretation",
        "positive": (
            r"emilia \(re:zero\), re:zero kara hajimeru isekai seikatsu, alternate costume, "
            "anime key visual, silver-white hair, very long hair, straight bangs, "
            "crown braid, symmetrical violet eyes, clear round irises, natural eye highlights, pointed elf ears, "
            "white rose hair ornament, x hair ornament, "
            "(pure white ethereal one-piece forest sundress:1.3), (soft lavender ribbon sash tied at waist:1.2), "
            "translucent white chiffon shawl over shoulders, green emerald crystal pendant at chest, "
            "(flowing layered white chiffon long skirt:1.2), delicate white lace stockings, white ankle boots"
        ),
        "negative_remove": ("floor-length gown, wedding dress, ",),
        "negative_add": (
            "heavy armor, dark purple bodysuit, modern clothes, official costume, default outfit, "
            "corset, waist panel, vertical purple dash markings, juliet bell sleeves, detached sleeves, "
            "split coattails, yellow hawk emblem, purple sleeve tails, miniskirt"
        ),
        "use_skirt_ipa": False,
        "use_brooch_detailer": False,
        "lora_strength": 0.0,
        "lora_strength_base": 0.0,
        "lora_strength_finish": 0.28,
        "ipadapter_weight": 0.28,
        "reference_mode": "paired",
    },
    "sanctuary": {
        "title": "圣域冬日主题创作服装（非官设复刻） / Winter interpretation",
        "positive": (
            f"{ULTRA_EMILIA_OUTFIT_CANONICAL}, "
            "translucent frost-trimmed white winter mantle draped over elbows, "
            "glowing emerald spirit crystal, subtle frost filigree on dress hem"
        ),
        "negative_remove": (),
        "negative_add": "hood, fur hood, heavy coat covering dress",
        "use_skirt_ipa": True,
        "use_brooch_detailer": True,
        "lora_strength": 0.25,
        "lora_strength_base": 0.25,
        "lora_strength_finish": 0.25,
        "ipadapter_weight": 0.18,
        "reference_mode": "single",
    },
    "nightgown": {
        "title": "宅邸晨衣主题创作服装（非官设复刻） / Morning dress interpretation",
        "positive": (
            r"emilia \(re:zero\), re:zero kara hajimeru isekai seikatsu, alternate costume, "
            "anime key visual, silver-white hair, very long loose hair, straight bangs, "
            "symmetrical violet eyes, clear round irises, natural eye highlights, pointed elf ears, "
            "white rose hair ornament, "
            "(loose flowing opaque white and soft lavender nightgown dress with ruffled collar and ribbon bow:1.3), "
            "(soft white morning robe draped over shoulders:1.2), green crystal pendant at throat, "
            "bare lower legs, soft bedroom morning light"
        ),
        "negative_remove": ("stocking feet, shoeless, barefoot, ", "floor-length gown, wedding dress, "),
        "negative_add": (
            "heavy boots, armor, hooded cloak, official costume, default outfit, "
            "corset, waist panel, vertical purple dash markings, bustier, juliet sleeves, detached sleeves, "
            "split coattails, yellow hawk emblem, thighhighs, stockings"
        ),
        "use_skirt_ipa": False,
        "use_brooch_detailer": False,
        "lora_strength": 0.0,
        "lora_strength_base": 0.0,
        "lora_strength_finish": 0.28,
        "ipadapter_weight": 0.28,
        "reference_mode": "paired",
    },
}

EMILIA_COMPANION_TAGS: dict[str, str] = {
    "none": "",
    "puck": (
        "tiny cute grey-furred fairy cat spirit Puck floating beside her shoulder, "
        "small grey kitten spirit with long fluffy tail and golden earring on left ear, "
        "warm glowing green spirit aura around the tiny cat"
    ),
}

EMILIA_MAGIC_TAGS: dict[str, str] = {
    "none": "",
    "ice": (
        "summoning translucent glowing cyan-blue ice crystals and hexagonal frost snowflakes in midair, "
        "crystalline mana particles, magical ice refraction, ethereal blue-violet rim light on fingers and dress"
    ),
    "blizzard": (
        "surrounded by floating diamond-dust ice shards and blooming translucent ice flowers, "
        "swirling frost magic aura, dramatic cold blue magical illumination"
    ),
}

EMILIA_ANGLE_SPECS: dict[str, dict[str, Any]] = {
    "auto": {
        "title": "智能视角匹配 (Auto Pose-Aware Multi-Ref)",
        "tags": "",
        "primary_ref": "official_face_center_clean.png",
        "secondary_ref": "emilia_official_face_3q.png",
    },
    "front": {
        "title": "正面标准视角 (Front Center)",
        "tags": "straight-on front view, symmetrical posture, looking directly at viewer",
        "primary_ref": "official_face_center_clean.png",
        "secondary_ref": "emilia_official_face_3q.png",
    },
    "three-quarter": {
        "title": "3/4 经典斜侧视角 (Three-Quarter View)",
        "tags": "three-quarter view, angled body posture, dynamic depth, looking at viewer",
        "primary_ref": "emilia_official_face_3q.png",
        "secondary_ref": "official_face_center_clean.png",
    },
    "side": {
        "title": "唯美侧颜回眸 (Side Profile / Looking Back)",
        "tags": "from side, graceful side profile turning head to look back at viewer, flowing silver hair over shoulder",
        "primary_ref": "emilia_official_face_3q.png",
        "secondary_ref": "official_face_center_clean.png",
    },
    "high-angle": {
        "title": "微俯视圣洁视角 (High-Angle Perspective)",
        "tags": "from slightly above, high-angle shot looking up at viewer with gentle expression",
        "primary_ref": "emilia_official_face_3q.png",
        "secondary_ref": "official_face_center_clean.png",
    },
    "low-angle": {
        "title": "微仰视史诗视角 (Heroic Low-Angle)",
        "tags": "from slightly below, low-angle heroic perspective, flowing dress coattails",
        "primary_ref": "emilia_official_face_3q.png",
        "secondary_ref": "official_face_center_clean.png",
    },
}

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
    "gentle smile, looking at viewer, full body, medium-long shot, subject fills the frame, "
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

# Built-in curated scene & lighting presets for one-command CLI invocation
BUILTIN_SCENE_PRESETS: dict[str, dict[str, Any]] = {
    "sunset-terrace": {
        "title": "黄昏宫殿露台 (Golden Sunset Terrace)",
        "framing": "full_body",
        "default_seed": 20261108,
        "tags": (
            "standing on an ornate marble palace terrace at sunset, three-quarter view, "
            "wind gently blowing silver hair and dress coattails, warm golden-orange sunset glow "
            "and cool twilight rim light, glowing floating ice crystals, serene confident smile, looking at viewer"
        ),
    },
    "crystal-sanctuary": {
        "title": "水晶森林圣域 (Enchanted Crystal Sanctuary)",
        "framing": "full_body",
        "default_seed": 20261109,
        "tags": (
            "walking gracefully through an enchanted crystal forest sanctuary, sunbeams piercing through canopy, "
            "dappled volumetric god rays, glowing blue mana particles and white petals floating in the air, "
            "gentle hand near chest, looking at viewer"
        ),
    },
    "moonlit-balcony": {
        "title": "月夜古堡露台 (Moonlit Castle Balcony)",
        "framing": "full_body",
        "default_seed": 20261104,
        "tags": (
            "standing on a grand stone balcony at night under a luminous full moon, "
            "warm golden lantern light from the left and cool silver moonlight from the right, "
            "summoning a translucent glowing ice crystal snowflake in one hand, starry night sky, "
            "gentle mystical smile, looking at viewer"
        ),
    },
    "rose-garden": {
        "title": "晴日白蔷薇庭院 (Sunlit White Rose Garden)",
        "framing": "full_body",
        "default_seed": 20261027,
        "tags": (
            "standing on a sunlit stone garden path, blooming white roses, stone archway, "
            "intricate background foliage, soft golden hour volumetric light, warm key light and cool ambient shadows, "
            "hands relaxed, gentle smile, looking at viewer"
        ),
    },
    "royal-portrait": {
        "title": "王选礼服半身特写 (Royal Selection Portrait)",
        "framing": "portrait",
        "default_seed": 20261105,
        "tags": (
            "close-up portrait, upper body, soft sunlit cathedral window background, "
            "delicate floating light particles, gentle warm smile, sparkling violet irises, "
            "luminous hair rim light, looking at viewer"
        ),
    },
    "snowy-capital": {
        "title": "王都初雪漫步 (First Snow in Royal Capital)",
        "framing": "full_body",
        "default_seed": 20261106,
        "tags": (
            "walking along a cobblestone street in a fantasy medieval town during gentle snowfall, "
            "warm shop window lights glowing in the background, soft winter twilight, "
            "holding a glowing ice crystal, gentle warm breath, looking back at viewer with a smile"
        ),
    },
}


def get_user_config_dir() -> Path:
    """Return the platform-appropriate global user config directory for Emilia Studio."""
    env_dir = os.environ.get("EMILIA_CONFIG_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "EmiliaStudio"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "emilia-studio"
    return Path.home() / ".config" / "emilia-studio"


def get_user_config_path() -> Path:
    """Return the default global config.toml file path."""
    env_cfg = os.environ.get("EMILIA_CONFIG_TOML")
    if env_cfg:
        return Path(env_cfg).expanduser().resolve()
    return get_user_config_dir() / "config.toml"


def get_bundled_resources_dir() -> Path | None:
    """Return read-only resources installed in the wheel or frozen application."""
    package_resources = Path(__file__).resolve().parent / "resources"
    if package_resources.is_dir():
        return package_resources
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / "emilia_bundled"
        if candidate.is_dir():
            return candidate
    return None


def get_source_checkout_dir() -> Path | None:
    """Recognize this package's checkout, never the caller's working directory."""
    if getattr(sys, "frozen", False):
        return None
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "pyproject.toml").is_file() and (candidate / "models_manifest.json").is_file():
        return candidate
    return None


def get_user_data_dir() -> Path:
    """Return a writable workspace without creating it during configuration reads."""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        return (Path(local) if local else Path.home() / "AppData" / "Local") / "EmiliaStudio"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "EmiliaStudio"
    data_home = os.environ.get("XDG_DATA_HOME")
    return (Path(data_home) if data_home else Path.home() / ".local" / "share") / "emilia-studio"


def _resolve_root_dir(explicit_root: str | Path | None = None) -> Path:
    root = explicit_root if explicit_root is not None else os.environ.get("EMILIA_ROOT_DIR")
    if root:
        return Path(root).expanduser().resolve()
    return get_source_checkout_dir() or get_user_data_dir()


def _external_python(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if getattr(sys, "frozen", False):
        resolved = shutil.which(str(candidate)) or candidate
        if Path(resolved).resolve() == Path(sys.executable).resolve():
            raise ValueError("conda_python must be an external Python interpreter, not the Emilia executable")
    return candidate


def _resolve_conda_python(root_dir: Path) -> Path | None:
    env_py = os.environ.get("EMILIA_CONDA_PYTHON")
    if env_py:
        return _external_python(env_py)
    executable = Path("python.exe") if sys.platform == "win32" else Path("bin/python")
    candidates = [root_dir / "env" / executable, root_dir / ".venv" / executable]
    if sys.platform == "win32":
        candidates.append(root_dir / ".venv" / "Scripts" / "python.exe")
    if not getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable))
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            try:
                return _external_python(candidate)
            except ValueError:
                continue
    # Keep unrelated commands usable; doctor/inference reports the missing Python.
    return None


def _resolve_ollama_exe() -> Path:
    env_ollama = os.environ.get("EMILIA_OLLAMA_EXE")
    if env_ollama:
        return Path(env_ollama).expanduser()
    which_ollama = shutil.which("ollama")
    if which_ollama:
        return Path(which_ollama)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.is_file():
            return candidate
    return Path("ollama")


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
    resources_dir: Path | None = None
    config_source: str = "auto-discovered"

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
    # Ultra profile LoRA. An explicit name, never a fallback chain: the same
    # arguments must build the same graph on every machine.
    ultra_lora_name: str = "Emilia_Rezero_Q-000005.safetensors"

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
    ultra_outfit_canonical: str = ULTRA_EMILIA_OUTFIT_CANONICAL
    ultra_positive_prompt: str = DEFAULT_ULTRA_POSITIVE_PROMPT
    ultra_negative_prompt: str = DEFAULT_ULTRA_NEGATIVE_PROMPT
    ultra_face_positive_prompt: str = DEFAULT_ULTRA_FACE_POSITIVE_PROMPT
    ultra_face_negative_prompt: str = DEFAULT_ULTRA_FACE_NEGATIVE_PROMPT
    ultra_hand_positive_prompt: str = DEFAULT_ULTRA_HAND_POSITIVE_PROMPT
    ultra_hand_negative_prompt: str = DEFAULT_ULTRA_HAND_NEGATIVE_PROMPT
    ultra_brooch_positive_prompt: str = DEFAULT_ULTRA_BROOCH_POSITIVE_PROMPT
    ultra_brooch_negative_prompt: str = DEFAULT_ULTRA_BROOCH_NEGATIVE_PROMPT

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir).expanduser().resolve()
        if self.resources_dir is None:
            env_resources = os.environ.get("EMILIA_RESOURCES_DIR")
            self.resources_dir = (
                Path(env_resources).expanduser().resolve() if env_resources else
                get_bundled_resources_dir() or get_source_checkout_dir() or
                Path(__file__).resolve().parent / "resources"
            )
        else:
            self.resources_dir = Path(self.resources_dir).expanduser().resolve()

        if self.app_dir is None:
            env_app = os.environ.get("EMILIA_APP_DIR")
            self.app_dir = Path(env_app).expanduser() if env_app else self.root_dir / "app"
        else:
            self.app_dir = Path(self.app_dir).expanduser()

        if self.output_dir is None:
            env_out = os.environ.get("EMILIA_OUTPUT_DIR")
            self.output_dir = Path(env_out).expanduser() if env_out else self.app_dir / "output"
        else:
            self.output_dir = Path(self.output_dir).expanduser()

        if self.temp_dir is None:
            env_temp = os.environ.get("EMILIA_TEMP_DIR")
            self.temp_dir = Path(env_temp).expanduser() if env_temp else self.root_dir / "temp"
        else:
            self.temp_dir = Path(self.temp_dir).expanduser()

        if self.workflows_dir is None:
            env_wf = os.environ.get("EMILIA_WORKFLOWS_DIR")
            if env_wf:
                self.workflows_dir = Path(env_wf).expanduser()
            else:
                self.workflows_dir = self.resources_dir / "workflows" / "api"
        else:
            self.workflows_dir = Path(self.workflows_dir).expanduser()

        if self.visual_workflows_dir is None:
            env_vwf = os.environ.get("EMILIA_VISUAL_WORKFLOWS_DIR")
            if env_vwf:
                self.visual_workflows_dir = Path(env_vwf).expanduser()
            else:
                self.visual_workflows_dir = self.resources_dir / "workflows" / "visual"
        else:
            self.visual_workflows_dir = Path(self.visual_workflows_dir).expanduser()
        if self.conda_python is None:
            self.conda_python = _resolve_conda_python(self.root_dir)
        else:
            self.conda_python = _external_python(self.conda_python)

        if self.ollama_exe is None:
            self.ollama_exe = _resolve_ollama_exe()
        else:
            self.ollama_exe = Path(self.ollama_exe).expanduser()

        for port_attr in ("comfy_port", "quality_port"):
            raw_port = getattr(self, port_attr)
            if isinstance(raw_port, bool):
                raise ValueError(f"{port_attr} must be an integer between 1 and 65535")
            port_val = int(raw_port)
            if not 1 <= port_val <= 65535:
                raise ValueError(f"{port_attr} must be an integer between 1 and 65535")
            setattr(self, port_attr, port_val)
    @property
    def comfy_base(self) -> str:
        """Base HTTP URL for standard ComfyUI service."""
        return f"http://{self.comfy_host}:{self.comfy_port}"

    @property
    def quality_base(self) -> str:
        """Base HTTP URL for isolated quality renderer ComfyUI service."""
        return f"http://{self.comfy_host}:{self.quality_port}"

    @property
    def manifest_path(self) -> Path:
        """Read-only model manifest shipped with this application."""
        return self.resources_dir / "models_manifest.json"

    @property
    def assets_dir(self) -> Path:
        """Read-only reference images shipped with this application."""
        return self.resources_dir / "assets"

    @property
    def patches_dir(self) -> Path:
        """Read-only runtime patches shipped with this application."""
        return self.resources_dir / "patches"

    @property
    def ledger_db_path(self) -> Path:
        """Writable ledger, preserving the source-checkout location."""
        return self.root_dir / "workflows" / "studio_ledger.sqlite3"

    def to_toml_string(self) -> str:
        """Serialize overrides without persisting frozen extraction directories."""
        paths = {
            "root_dir": self.root_dir,
            "app_dir": self.app_dir,
            "output_dir": self.output_dir,
            "temp_dir": self.temp_dir,
        }
        default_resources = (
            get_bundled_resources_dir() or get_source_checkout_dir() or
            Path(__file__).resolve().parent / "resources"
        )
        if self.resources_dir != default_resources:
            paths["resources_dir"] = self.resources_dir
        if self.workflows_dir != self.resources_dir / "workflows" / "api":
            paths["workflows_dir"] = self.workflows_dir
        if self.visual_workflows_dir != self.resources_dir / "workflows" / "visual":
            paths["visual_workflows_dir"] = self.visual_workflows_dir
        for key in ("conda_python", "ollama_exe"):
            value = getattr(self, key)
            resolved = shutil.which(str(value)) if value is not None else None
            if resolved or (value is not None and value.is_absolute()):
                paths[key] = Path(resolved) if resolved else value
        lines = [
            "# Emilia Studio configuration; relative paths resolve beside this file.",
            "[paths]",
        ]
        for key, value in paths.items():
            lines.append(f"{key} = {json.dumps(str(value).replace(chr(92), '/'), ensure_ascii=False)}")
        lines.extend([
            "", "[services]",
            f"comfy_host = {json.dumps(self.comfy_host, ensure_ascii=False)}",
            f"comfy_port = {self.comfy_port}",
            f"quality_port = {self.quality_port}",
            f"ollama_base = {json.dumps(self.ollama_base, ensure_ascii=False)}",
            f"ollama_model = {json.dumps(self.ollama_model, ensure_ascii=False)}",
            "",
        ])
        default_lora = type(self).__dataclass_fields__["ultra_lora_name"].default
        if self.ultra_lora_name != default_lora:
            lines.extend(["[models]", f"ultra_lora_name = {json.dumps(self.ultra_lora_name, ensure_ascii=False)}", ""])
        return "\n".join(lines)

    def save_user_config(self, target_path: str | Path | None = None) -> Path:
        """Persist current configuration to global user config.toml."""
        out_path = Path(target_path) if target_path is not None else get_user_config_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.to_toml_string(), encoding="utf-8")
        return out_path


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
    """Load explicit/env config, then user config, then this checkout's config.

    Environment values override TOML values. Relative TOML paths are anchored
    to the selected file, not the process's current working directory.
    """
    selected = config_path if config_path is not None else os.environ.get("EMILIA_CONFIG_TOML")
    if selected is None:
        user_config = get_user_config_dir() / "config.toml"
        source = get_source_checkout_dir()
        if user_config.is_file():
            selected = user_config
        elif source is not None and (source / "config.toml").is_file():
            selected = source / "config.toml"

    cfg_file = Path(selected).expanduser().resolve() if selected is not None else None
    overrides: dict[str, Any] = {}
    if cfg_file is not None:
        # Explicit missing or malformed configuration must not silently fall back.
        with cfg_file.open("rb") as stream:
            overrides = _flatten_toml(tomllib.load(stream))

    kwargs: dict[str, Any] = {"config_source": str(cfg_file) if cfg_file else "auto-discovered"}
    path_fields = {
        "root_dir", "app_dir", "output_dir", "temp_dir", "workflows_dir",
        "visual_workflows_dir", "conda_python", "ollama_exe", "resources_dir",
    }
    env_map = {
        "root_dir": "EMILIA_ROOT_DIR",
        "app_dir": "EMILIA_APP_DIR",
        "output_dir": "EMILIA_OUTPUT_DIR",
        "temp_dir": "EMILIA_TEMP_DIR",
        "workflows_dir": "EMILIA_WORKFLOWS_DIR",
        "visual_workflows_dir": "EMILIA_VISUAL_WORKFLOWS_DIR",
        "resources_dir": "EMILIA_RESOURCES_DIR",
        "conda_python": "EMILIA_CONDA_PYTHON",
        "ollama_exe": "EMILIA_OLLAMA_EXE",
        "comfy_host": "EMILIA_COMFY_HOST",
        "comfy_port": "EMILIA_COMFY_PORT",
        "quality_port": "EMILIA_QUALITY_PORT",
        "ollama_base": "EMILIA_OLLAMA_BASE",
        "ollama_model": "EMILIA_OLLAMA_MODEL",
    }
    for key, value in overrides.items():
        if key not in StudioConfig.__dataclass_fields__ or key == "config_source":
            continue
        if key in path_fields:
            if not value:
                continue
            value = Path(value).expanduser()
            if not value.is_absolute():
                value = cfg_file.parent / value
            value = value.resolve()
        kwargs[key] = value
    for key, variable in env_map.items():
        value = os.environ.get(variable)
        if value:
            kwargs[key] = Path(value).expanduser() if key in path_fields else value
    return StudioConfig(**kwargs)
