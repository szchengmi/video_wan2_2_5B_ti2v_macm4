#!/usr/bin/env python3
"""
Step 4: 视频生成 - Wan2.2 TI2V 5B (fp16 safetensors)
=====================================================
通过 ComfyUI API 调用 Wan2.2 生成视频。
工作流: UNETLoader → ModelSamplingSD3 → Wan22ImageToVideoLatent → KSampler → VAEDecode → VHS_VideoCombine

模型: /kaggle/input/saysnkaggle/wan2-2-5b-f16/models/
  - wan2.2_ti2v_5B_fp16.safetensors (UNET, ~10GB)
  - umt5_xxl_fp8_e4m3fn_scaled.safetensors (CLIP, ~6.7GB)
  - wan2.2_vae.safetensors (VAE, ~1.4GB)
"""

import os
import sys
import json
import time
import shutil
import urllib.request
import urllib.error
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    log, save_json, load_json, run_cmd,
    EPISODE_NUM, get_dirs, WAN22_MODELS_DIR,
    COMFYUI_GPU_FLAG, _IS_APPLE_SILICON, _IS_MAC,
)

# ============================================================
# ComfyUI 管理 (复用 wan22-ai-series 的成熟逻辑)
# ============================================================

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_DIR = "/kaggle/working/ComfyUI"


def _find_comfyui():
    """查找 ComfyUI 安装位置"""
    try:
        import importlib.util
        spec = importlib.util.find_spec("comfy")
        if spec and spec.origin:
            return ("pip", os.path.dirname(spec.origin))
    except:
        pass
    for candidate in ["/kaggle/working/ComfyUI", "/kaggle/working/ComfyUI-master"]:
        if os.path.isdir(candidate) and os.path.isfile(f"{candidate}/main.py"):
            return ("dir", candidate)
    return None


def _install_comfyui():
    """安装 ComfyUI + 必要插件"""
    run_cmd("pkill -f 'main.py' 2>/dev/null; true")
    if os.path.isdir(COMFYUI_DIR):
        shutil.rmtree(COMFYUI_DIR, ignore_errors=True)
    log("安装 ComfyUI...")
    run_cmd(f"git clone https://github.com/Comfyanonymous/ComfyUI.git {COMFYUI_DIR}", timeout=120)
    os.chdir(COMFYUI_DIR)
    run_cmd("pip install -r requirements.txt 2>&1 | tail -3", timeout=300)
    # 安装插件
    cn_dir = f"{COMFYUI_DIR}/custom_nodes"
    os.makedirs(cn_dir, exist_ok=True)
    if not os.path.isdir(f"{cn_dir}/ComfyUI-VideoHelperSuite"):
        log("  安装 VideoHelperSuite...")
        run_cmd(f"git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git {cn_dir}/ComfyUI-VideoHelperSuite", timeout=60)
        req = f"{cn_dir}/ComfyUI-VideoHelperSuite/requirements.txt"
        if os.path.isfile(req):
            run_cmd(f"pip install -r {req} 2>&1 | tail -3", timeout=60)


def _create_extra_model_paths():
    """创建 extra_model_paths.yaml 注册模型路径"""
    if _IS_MAC:
        # Mac 上直接用 models/ 下的子目录
        yaml_content = f"""\
wan22_ti2v:
  base_path: {WAN22_MODELS_DIR}
  diffusion_models: ./unet
  text_encoders: ./clip
  vae: ./vae
"""
    else:
        yaml_content = f"""\
wan22_ti2v:
  base_path: {WAN22_MODELS_DIR}
  diffusion_models: .
  text_encoders: .
  vae: .
"""
    yaml_path = f"{COMFYUI_DIR}/extra_model_paths.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    log(f"  extra_model_paths.yaml: {yaml_path}")


def _start_comfyui():
    """启动 ComfyUI 服务器"""
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=2)
        log("ComfyUI 已在运行")
        return True
    except:
        pass

    result = _find_comfyui()
    if result:
        install_type, path = result
        log(f"  ComfyUI ({install_type}): {path}")
    else:
        _install_comfyui()
        result = _find_comfyui()
        if not result:
            log("❌ ComfyUI 安装失败")
            return False

    _create_extra_model_paths()

    cwd = result[1] if isinstance(result[1], str) and os.path.isdir(result[1]) else None
    log_file = "/kaggle/working/ai-series/comfyui.log"
    log_fh = open(log_file, "w")

    has_gpu = False
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except:
        pass
    gpu_flag = COMFYUI_GPU_FLAG
    if has_gpu:
        log(f"  GPU: MPS (Apple Silicon)" if _IS_APPLE_SILICON else f"  GPU: CUDA")
    else:
        log(f"  GPU: CPU模式")
    log(f"  ComfyUI 启动参数: {gpu_flag}")

    env = os.environ.copy()
    if not _IS_APPLE_SILICON:
        env["CUDA_VISIBLE_DEVICES"] = "0"

    if result[0] == "pip":
        proc = subprocess.Popen(
            ["python", "-m", "comfy", "main", "--dont-print-server",
             "--preview-method", "none", "--listen", "0.0.0.0", "--port", "8188"] + gpu_flag,
            stdout=log_fh, stderr=subprocess.STDOUT, env=env,
        )
    else:
        proc = subprocess.Popen(
            ["python", "main.py", "--dont-print-server",
             "--preview-method", "none", "--listen", "0.0.0.0", "--port", "8188"] + gpu_flag,
            cwd=cwd, stdout=log_fh, stderr=subprocess.STDOUT, env=env,
        )

    for i in range(200):
        time.sleep(3)
        try:
            urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=2)
            log(f"  ComfyUI 就绪 ({(i+1)*3}s)")
            log_fh.close()
            return True
        except:
            if i < 5 or i % 20 == 0:
                log(f"  ⏳ 等待... ({(i+1)*3}s)")

    log_fh.close()
    log("  ❌ ComfyUI 启动超时")
    return False


# ============================================================
# 模型查找
# ============================================================

def _find_wan22_models():
    """查找 Wan2.2 模型文件，兼容 Mac 本地和 Kaggle 路径"""
    search_paths = [WAN22_MODELS_DIR]

    # Mac 本地: 模型在 models/unet/, models/vae/, models/clip/
    if _IS_MAC:
        for sub in ["unet", "vae", "clip"]:
            p = os.path.join(WAN22_MODELS_DIR, sub)
            if os.path.isdir(p):
                search_paths.append(p)
        # 也搜索 models/ 下直接放文件的情况
        if os.path.isdir(WAN22_MODELS_DIR):
            search_paths.append(WAN22_MODELS_DIR)

    # Kaggle: 也搜索 Dataset 的 models/ 子目录
    if _IS_KAGGLE:
        for d in ["/kaggle/input/saysnkaggle/wan2-2-5b-f16",
                  "/kaggle/input/saysnkaggle/wan2-2-5b-f16/models"]:
            if os.path.isdir(d):
                search_paths.append(d)

    result = {"unet": None, "clip": None, "vae": None}
    for base in search_paths:
        if not os.path.isdir(base):
            continue
        for f in os.listdir(base):
            fl = f.lower()
            fp = os.path.join(base, f)
            if not os.path.isfile(fp):
                continue
            if result["unet"] is None and "wan2.2_ti2v_5b" in fl and "fp16" in fl:
                result["unet"] = fp
            elif result["clip"] is None and "umt5_xxl" in fl:
                result["clip"] = fp
            elif result["vae"] is None and "wan2.2_vae" in fl:
                result["vae"] = fp
    return result


# ============================================================
# ComfyUI API
# ============================================================

def _queue_prompt(workflow):
    """提交工作流"""
    payload = json.dumps({"prompt": workflow, "client_id": "wan22-ti2v"}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        log(f"  HTTP {e.code}: {body[:300]}")
        raise


def _wait_for_completion(prompt_id, timeout=1800):
    """等待工作流完成"""
    start = time.time()
    last_log = 0
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{COMFYUI_URL}/history/{prompt_id}")
            with urllib.request.urlopen(req, timeout=60) as resp:
                history = json.loads(resp.read())
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {}).get("status_str")
                if status == "success":
                    return entry
                if status == "error":
                    raise RuntimeError(f"工作流失败: {entry}")
                elapsed = time.time() - start
                if elapsed - last_log >= 30:
                    log(f"    ⏳ {elapsed:.0f}s")
                    last_log = elapsed
        except urllib.error.HTTPError:
            pass
        time.sleep(5)
    raise TimeoutError(f"超时 ({timeout}s)")


# ============================================================
# 工作流构建 — 使用 ModelSamplingSD3 + shift 参数
# ============================================================

def _build_wan22_workflow(positive_prompt, negative_prompt,
                           unet_path, clip_path, vae_path,
                           width, height, frames,
                           steps, cfg, sampler, scheduler, shift, seed):
    """
    Wan2.2 TI2V 工作流 (fp16 safetensors)

    节点连接:
      1: UNETLoader → 6: ModelSamplingSD3.model
      2: CLIPLoader → 4,5: CLIPTextEncode.clip
      3: VAELoader → 7: Wan22ImageToVideoLatent.vae, 9: VAEDecode.vae
      4: CLIPTextEncode (positive) → 7: Wan22ImageToVideoLatent
      5: CLIPTextEncode (negative) → 7: Wan22ImageToVideoLatent
      6: ModelSamplingSD3 → 8: KSampler.model  (★ shift 参数在这里)
      7: Wan22ImageToVideoLatent → 8: KSampler.latent_image
      8: KSampler → 9: VAEDecode.samples
      9: VAEDecode → 10: VHS_VideoCombine.images
      10: VHS_VideoCombine → 输出视频

    关键: ModelSamplingSD3 将 shift 参数应用到 UNET 模型
    """
    unet_name = os.path.basename(unet_path)
    clip_name = os.path.basename(clip_path)
    vae_name = os.path.basename(vae_path)

    # Wan22ImageToVideoLatent 的 latent 参数
    # 48-channel latent, 16x spatial downsample
    lat_h = height // 16
    lat_w = width // 16
    lat_frames = (frames - 1) // 4 + 1

    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"}
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "wan", "device": "default"}
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name}
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["2", 0]}
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]}
        },
        "6": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["1", 0],
                "shift": shift,
            }
        },
        "7": {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": ["3", 0],
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1,
            }
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["6", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            }
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["3", 0],
            }
        },
        "10": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["9", 0],
                "frame_rate": 8,
                "loop_count": 0,
                "filename_prefix": "wan22",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            }
        },
    }
    return workflow


def _save_placeholder_video(shot, output_path, num_frames):
    """生成占位视频"""
    from PIL import Image, ImageDraw
    res = 480
    img = Image.new("RGB", (res, res), (20, 20, 40))
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "[VIDEO]", fill=(200, 200, 255))
    draw.text((20, 70), shot.get("shot_id", ""), fill=(200, 255, 200))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp = output_path.replace(".mp4", "_tmp.png")
    img.save(tmp)
    dur = max(num_frames / 8, 1)
    run_cmd(f'ffmpeg -y -loop 1 -i "{tmp}" -t {dur} -c:v libx264 -pix_fmt yuv420p -movflags +faststart "{output_path}" 2>/dev/null')
    if os.path.exists(tmp):
        os.remove(tmp)


# ============================================================
# 主函数
# ============================================================

def main(storyboard=None):
    log("=" * 50)
    log("Step 4: 视频生成 (Wan2.2 TI2V 5B fp16)")
    log("=" * 50)

    dirs = get_dirs(EPISODE_NUM)
    total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))

    # 模型
    models = _find_wan22_models()
    for name, path in [("UNET", models["unet"]), ("CLIP", models["clip"]), ("VAE", models["vae"])]:
        if not path:
            log(f"  ❌ {name} 未找到")
            return
        log(f"  ✅ {name}: {os.path.basename(path)} ({os.path.getsize(path)/1e6:.0f}MB)")

    # ComfyUI
    if not _start_comfyui():
        log("❌ ComfyUI 启动失败")
        return

    # 参数 (对齐参考工作流的 shift=8, euler/simple)
    w, h = 832, 480
    num_frames = 49  # ~6s @ 8fps → 13 latent 帧
    steps = 20
    cfg = 5.0
    sampler = "euler"
    scheduler = "simple"
    shift = 8.0

    style_name = storyboard.get("style", "未指定")
    log(f"参数: {w}x{h} | {num_frames}f | {steps}步 | CFG={cfg} | shift={shift} | {sampler}/{scheduler} | 风格: {style_name}")

    count = 0
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            ep = storyboard.get("episode", 1)
            out = f"{dirs['videos']}/ep{ep:02d}_{scene['scene_id']}_{sid}.mp4"

            if os.path.exists(out) and os.path.getsize(out) > 100000:
                log(f"  [{count}/{total}] {sid} 跳过")
                continue

            prompt_base = shot.get('prompt', '')
            if 'anime style' not in prompt_base and 'style' not in storyboard:
                # 兼容旧 storyboard 无风格字段的情况，追加默认后缀
                video_prompt = f"{prompt_base}, smooth motion, cinematic, high quality"
            else:
                video_prompt = f"{prompt_base}, smooth motion, cinematic"
            neg_prompt = shot.get("negative_prompt", "blurry, distorted, static, motionless, low quality")
            seed = shot.get("seed", 42)
            if isinstance(seed, str) or seed < 0:
                seed = 42

            workflow = _build_wan22_workflow(
                positive_prompt=video_prompt,
                negative_prompt=neg_prompt,
                unet_path=models["unet"],
                clip_path=models["clip"],
                vae_path=models["vae"],
                width=w, height=h, frames=num_frames,
                steps=steps, cfg=cfg,
                sampler=sampler, scheduler=scheduler,
                shift=shift, seed=seed,
            )

            try:
                log(f"  [{count}/{total}] {sid} 提交...")
                result = _queue_prompt(workflow)
                prompt_id = result.get("prompt_id")
                if not prompt_id:
                    raise RuntimeError(f"无 prompt_id: {result}")

                completion = _wait_for_completion(prompt_id, timeout=1800)
                outputs = completion.get("outputs", {})

                video_output = None
                for nid, nout in outputs.items():
                    if "gifs" in nout:
                        gifs = nout["gifs"]
                        if isinstance(gifs, list) and gifs:
                            video_output = gifs[0]
                            if isinstance(video_output, dict):
                                video_output = video_output.get("fullpath", "")
                            break

                if video_output and os.path.isfile(video_output):
                    shutil.copy2(video_output, out)
                    size_mb = os.path.getsize(out) / 1e6
                    log(f"  [{count}/{total}] {sid} ✓ ({size_mb:.1f}MB)")
                    if size_mb < 0.05:
                        log(f"    ⚠️ 文件极小，可能无运动")
                else:
                    log(f"  [{count}/{total}] {sid} 无输出")
                    _save_placeholder_video(shot, out, num_frames)

            except Exception as e:
                log(f"  [{count}/{total}] {sid} 失败: {e}")
                _save_placeholder_video(shot, out, num_frames)

    log("视频生成完成")


if __name__ == "__main__":
    dirs = get_dirs(EPISODE_NUM)
    storyboard = load_json(f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json")
    main(storyboard)
