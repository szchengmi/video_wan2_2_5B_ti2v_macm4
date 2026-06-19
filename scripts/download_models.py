#!/usr/bin/env python3
"""
Wan2.2 TI2V 5B 模型下载脚本 (Mac M4 / Kaggle 通用)
只下载 pipeline 实际需要的 3 个模型文件:
  - wan2.2_ti2v_5B_fp16.safetensors (UNET, ~10GB)
  - umt5_xxl_fp8_e4m3fn_scaled.safetensors (CLIP, ~12GB)
  - wan2.2_vae.safetensors (VAE, ~1.5GB)
"""

import os
import sys
import time
import shutil

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_HUB_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print(f"[OK] HF_TOKEN: {HF_TOKEN[:10]}...")
else:
    print("[WARN] HF_TOKEN 未设置，私有模型可能下载失败")

# 默认模型目录
DEFAULT_MODEL_DIR = os.path.expanduser("~/heipiworkspace/mac/projects/video_wan2_2_5B_ti2v_macm4/models")
MODEL_DIR = os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)

# HuggingFace 镜像 (国内加速)
HF_MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com")


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_disk_free_gb(path="/"):
    import shutil as _s
    _, _, free = _s.disk_usage(path)
    return free / 1e9


def download_hf(repo_id, filename, dest_path, use_mirror=True):
    """从 HuggingFace 下载单个文件"""
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    mirrors_url = None
    if use_mirror:
        mirror_repo = repo_id.replace("/", ".")

    try:
        result = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.dirname(dest_path),
            token=HF_TOKEN or None,
        )
        return True
    except Exception as e:
        log(f"    HF 直链失败: {e}")

    # 尝试镜像
    if use_mirror:
        try:
            from huggingface_hub import hf_hub_download
            mirror_repo = f"hf-mirror.com/{repo_id}"
            result = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=os.path.dirname(dest_path),
                token=HF_TOKEN or None,
                endpoint=HF_MIRROR,
            )
            return True
        except:
            pass

    # 最终 fallback: wget 镜像
    if use_mirror:
        url = f"{HF_MIRROR}/{repo_id}/resolve/main/{filename}"
        log(f"    尝试 wget 镜像: {url}")
        import subprocess
        try:
            subprocess.run(
                ["wget", "-q", "--show-progress", "-O", dest_path, url],
                timeout=3600,
            )
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
                return True
        except:
            pass

    return False


def download_wget(url, dest_path):
    """wget 下载"""
    import subprocess
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    log(f"    wget: {url[:80]}...")
    try:
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", dest_path, url],
            timeout=3600,
        )
        return os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024
    except:
        return False


# ============================================================
# 模型定义 — Wan2.2 TI2V 5B 完整 pipeline
# ============================================================

MODELS = [
    {
        "id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "name": "Wan2.2 UNET 5B",
        "dir": "unet",
        "desc": "~10GB",
        "files": [
            "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        ],
    },
    {
        "id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "name": "Wan2.2 VAE",
        "dir": "vae",
        "desc": "~1.5GB",
        "files": [
            "split_files/vae/wan2.2_vae.safetensors",
        ],
    },
    {
        "id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "name": "Wan2.2 CLIP (umt5)",
        "dir": "clip",
        "desc": "~12GB",
        "files": [
            "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        ],
    },
]


def main():
    log("=" * 55)
    log("  Wan2.2 TI2V 5B 模型下载")
    log("=" * 55)
    log(f"目标目录: {MODEL_DIR}")
    log(f"磁盘剩余: {get_disk_free_gb():.1f}GB")

    os.makedirs(MODEL_DIR, exist_ok=True)

    subprocess.run("pip install -q -U huggingface_hub", shell=True, timeout=120)

    for i, model in enumerate(MODELS, 1):
        log(f"\n{'='*55}")
        log(f"[{i}/{len(MODELS)}] {model['name']} ({model['desc']})")

        target = os.path.join(MODEL_DIR, model["dir"])

        # 检查是否已存在
        all_exist = True
        for filename in model["files"]:
            dest = os.path.join(target, filename)
            if not os.path.exists(dest) or os.path.getsize(dest) < 100 * 1024 * 1024:
                all_exist = False
                break

        if all_exist:
            log(f"  ✅ 已存在，跳过")
            continue

        os.makedirs(target, exist_ok=True)
        t0 = time.time()

        ok = 0
        for filename in model["files"]:
            dest = os.path.join(target, filename)
            if os.path.exists(dest) and os.path.getsize(dest) > 100 * 1024 * 1024:
                ok += 1
                log(f"  ✅ {filename} (已存在)")
                continue

            log(f"  ⬇️  {filename}")
            success = download_hf(model["id"], filename, dest)

            if success and os.path.exists(dest):
                size_mb = os.path.getsize(dest) / 1e6
                ok += 1
                log(f"  ✅ {filename} ({size_mb:.0f}MB)")
            else:
                log(f"  ❌ {filename} 下载失败")

            free = get_disk_free_gb()
            if free < 0.5:
                log(f"  ⚠️  磁盘不足！剩余{free:.1f}GB")
                break

        elapsed = time.time() - t0
        if ok == len(model["files"]):
            log(f"  ✅ {model['name']} 完成 ({elapsed:.0f}秒)")
        else:
            log(f"  ⚠️  {model['name']} 部分完成 ({ok}/{len(model['files'])})")

    # 最终结果
    log(f"\n{'='*55}")
    log("全部完成！")
    total = sum(
        os.path.getsize(os.path.join(MODEL_DIR, f))
        for root, _, files in os.walk(MODEL_DIR)
        for f in files
        if os.path.isfile(os.path.join(root, f))
    )
    free = get_disk_free_gb()
    log(f"模型总计: {total/1e9:.2f}GB | 磁盘剩余: {free:.1f}GB")
    log("=" * 55)


if __name__ == "__main__":
    main()
