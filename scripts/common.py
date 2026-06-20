"""
公共配置和工具函数 - 所有step脚本共享
  自动检测平台: Kaggle (T4) / Mac (MPS) / 其他 (CPU)
"""

import os
import sys
import json
import time
import platform
import subprocess

# ============================================================
# 平台检测
# ============================================================

_IS_KAGGLE = os.path.isdir("/kaggle/input")
_IS_MAC = platform.system() == "Darwin"
_IS_APPLE_SILICON = _IS_MAC and platform.machine() == "arm64"

# MPS 支持: Apple Silicon 上启用
if _IS_APPLE_SILICON:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ============================================================
# 默认配置（可被环境变量覆盖）
# ============================================================

def _get_kaggle_secret(key_name):
    """从 Kaggle Secrets 读取密钥（兼容 Notebook 预赋值变量）"""
    # 方式1: Notebook 里已赋值的变量（如 secret_value_1）
    try:
        val = eval(key_name)
        if val:
            return val
    except:
        pass
    # 方式2: kaggle_secrets 库
    try:
        from kaggle_secrets import UserSecretsClient
        val = UserSecretsClient().get_secret(key_name)
        if val:
            return val
    except:
        pass
    # 方式3: 环境变量
    return os.environ.get(key_name, "")

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# 平台相关默认路径
if _IS_KAGGLE:
    BASE_DIR = os.environ.get("BASE_DIR", "/kaggle/working/ai-series")
    WAN22_DATASET = os.environ.get("WAN22_DATASET", "/kaggle/input/datasets/saysnkaggle/wan2-2-5b-f16")
elif _IS_MAC:
    _MAC_DEFAULT = "/Users/heipi/ComfyUI/models"
    BASE_DIR = os.environ.get("BASE_DIR", os.path.expanduser("~/heipiworkspace/mac/projects/video_wan2_2_5B_ti2v_macm4/output"))
    WAN22_DATASET = os.environ.get("WAN22_DATASET", _MAC_DEFAULT)
else:
    BASE_DIR = os.environ.get("BASE_DIR", os.path.expanduser("~/output"))
    WAN22_DATASET = os.environ.get("WAN22_DATASET", os.path.expanduser("~/models"))

WAN22_MODELS_DIR = WAN22_DATASET  # Mac 上模型文件直接在目录下

EPISODE_NUM = int(os.environ.get("EPISODE_NUM", "1"))

# ComfyUI 启动参数
if _IS_APPLE_SILICON:
    COMFYUI_GPU_FLAG = ["--force-mps"]
elif not _IS_KAGGLE:
    COMFYUI_GPU_FLAG = []  # 本地非 Mac 默认 CPU
else:
    COMFYUI_GPU_FLAG = []  # Kaggle 自动检测

# 视频参数 (Wan2.2 TI2V 5B)
WAN22_WIDTH = 832
WAN22_HEIGHT = 480
WAN22_FRAMES = 49   # ~6s @ 8fps
WAN22_FPS = 8
WAN22_STEPS = 20
WAN22_CFG = 5.0
WAN22_SAMPLER = "euler"
WAN22_SCHEDULER = "simple"
WAN22_SHIFT = 8.0
WAN22_DENOISE = 1.0

# 质量预设
QUALITY_MODE = os.environ.get("QUALITY_MODE", "fast")
QUALITY_PRESETS = {
    "fast": {"steps": 15, "resolution": 512, "fps": 8},
    "balanced": {"steps": 20, "resolution": 512, "fps": 8},
    "quality": {"steps": 30, "resolution": 768, "fps": 12},
}
PRESET = QUALITY_PRESETS.get(QUALITY_MODE, QUALITY_PRESETS["fast"])

IMAGE_STEPS = int(os.environ.get("IMAGE_STEPS", str(PRESET["steps"])))
IMAGE_GUIDANCE = float(os.environ.get("IMAGE_GUIDANCE", "7.5"))
VIDEO_FPS = int(os.environ.get("VIDEO_FPS", str(PRESET["fps"])))
VIDEO_RESOLUTION = int(os.environ.get("VIDEO_RESOLUTION", str(PRESET["resolution"])))
AUDIO_SAMPLE_RATE = 24000

# HuggingFace
if HF_TOKEN:
    os.environ["HF_HUB_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


# ============================================================
# 目录结构
# ============================================================

def get_dirs(episode_num=EPISODE_NUM):
    """获取目录结构"""
    ep_dir = f"{BASE_DIR}/episode_{episode_num:02d}"
    return {
        "base": BASE_DIR,
        "episode": ep_dir,
        "storyboard": f"{ep_dir}/storyboards",
        "images": f"{ep_dir}/images",
        "videos": f"{ep_dir}/videos",
        "audio": f"{ep_dir}/audio",
        "final": f"{ep_dir}/final",
        "models": f"{BASE_DIR}/models",
        "logs": f"{ep_dir}/logs",
    }


def setup_dirs(episode_num=EPISODE_NUM):
    """创建目录"""
    dirs = get_dirs(episode_num)
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


# ============================================================
# 工具函数
# ============================================================

def log(msg):
    """带时间戳的日志"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_cmd(cmd, timeout=600):
    """执行shell命令"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return result


def save_json(data, path):
    """保存JSON文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    """加载JSON文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def seconds_to_srt_time(seconds):
    """秒转SRT时间格式 HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ============================================================
# 角色/场景设定
# ============================================================

CHARACTER_PROMPTS = {
    "xiaoming": {
        "base_prompt": "1boy, young Chinese man, short black hair, wearing glasses, wearing dark hoodie, anime style, high quality",
        "negative_prompt": "ugly, deformed, bad anatomy, blurry, low quality",
        "seed": 42
    },
    "xiaoli": {
        "base_prompt": "1girl, young Chinese woman, long black hair, wearing light-colored dress, anime style, high quality",
        "negative_prompt": "ugly, deformed, bad anatomy, blurry, low quality",
        "seed": 123
    },
    "boss_wang": {
        "base_prompt": "1man, middle-aged Chinese man, square face, thick eyebrows, wearing business suit, anime style, high quality",
        "negative_prompt": "ugly, deformed, bad anatomy, blurry, low quality",
        "seed": 456
    }
}

SCENE_PROMPTS = {
    "office": "modern office interior, floor-to-ceiling windows, minimalist design, warm tones, anime background",
    "cafe": "cozy cafe interior, wooden furniture, warm yellow lighting, anime background",
    "park": "city park, green trees, bench and fountain, anime background",
    "apartment": "cozy apartment, Nordic style, living room, anime background",
    "street": "city street at dusk, street lamps, anime background"
}

EMOTION_ENHANCE = {
    "happy": "smiling, bright expression",
    "sad": "sad expression, teary eyes",
    "angry": "angry expression, furrowed brows",
    "surprised": "surprised expression, wide eyes",
    "nervous": "nervous expression, sweating",
    "calm": "calm expression, relaxed",
    "determined": "determined expression, confident",
    "embarrassed": "embarrassed expression, blushing",
    "thoughtful": "thoughtful expression, contemplative"
}

# ============================================================
# 风格预设 (Style Presets)
# ============================================================

STYLE_PRESETS = {
    "二次元": {
        "prompt_suffix": "anime style, vibrant colors, clean lines, studio ghibli inspired, detailed eyes, dynamic composition, cel shading",
        "negative_prompt": "realistic, photographic, 3D render, western cartoon, ugly, deformed, blurry",
        "scene_suffix": "anime background",
    },
    "古代田园": {
        "prompt_suffix": "ancient Chinese pastoral style, traditional ink painting, rice paddies, misty mountains, soft watercolor, warm earth tones, handscroll aesthetic",
        "negative_prompt": "modern, city, technology, neon, cars, buildings, anime, cartoon, ugly, deformed",
        "scene_suffix": "ancient Chinese landscape, ink wash painting background",
    },
    "赛博朋克": {
        "prompt_suffix": "cyberpunk style, neon lights, dark city, rain, holographic displays, high contrast, blade runner aesthetic, volumetric lighting",
        "negative_prompt": "nature, daylight, pastel, soft, vintage, anime, cartoon, ugly, deformed",
        "scene_suffix": "cyberpunk city interior, neon lights, high-tech, dark atmosphere",
    },
    "动漫": {
        "prompt_suffix": "anime style, detailed shading, japanese animation style, expressive characters, vivid colors, key visual",
        "negative_prompt": "realistic, photographic, western cartoon, chibi, ugly, deformed, blurry",
        "scene_suffix": "anime background",
    },
    "类真人": {
        "prompt_suffix": "live-action style, photorealistic, cinematic lighting, shallow depth of field, film grain, natural skin texture, dramatic lighting",
        "negative_prompt": "anime, cartoon, illustration, drawing, painting, deformed, blurry, low quality",
        "scene_suffix": "realistic interior, cinematic lighting, photorealistic environment",
    },
    "火柴人": {
        "prompt_suffix": "stick figure style, minimalist, black lines on white background, simple shapes, clean design, no shading, line art",
        "negative_prompt": "detailed, realistic, colorful, complex, shaded, 3D, anime, ugly, deformed",
        "scene_suffix": "minimalist background, white background, simple lines",
    },
}

DEFAULT_STYLE = "二次元"

SHOT_PARAMS = {
    "close_up": {"w": 768, "h": 768, "prefix": "close-up shot of"},
    "medium_shot": {"w": 768, "h": 512, "prefix": "medium shot of"},
    "wide_shot": {"w": 1024, "h": 576, "prefix": "wide shot of"},
    "extreme_close_up": {"w": 512, "h": 768, "prefix": "extreme close-up of"}
}

VOICE_PARAMS = {
    "xiaoming": {"speed": 0.9, "temp": 0.3, "top_p": 0.7, "top_k": 20},
    "xiaoli": {"speed": 1.1, "temp": 0.4, "top_p": 0.8, "top_k": 25},
    "boss_wang": {"speed": 0.85, "temp": 0.25, "top_p": 0.6, "top_k": 15},
    "narrator": {"speed": 1.0, "temp": 0.3, "top_p": 0.7, "top_k": 20}
}

EMOTION_SPEED = {
    "happy": 1.1, "sad": 0.85, "angry": 1.15, "surprised": 1.2,
    "nervous": 1.1, "calm": 1.0, "determined": 1.05,
    "embarrassed": 1.1, "thoughtful": 0.9
}
