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

# 仅对本地 ComfyUI 绕过代理，Gemini API 等外网继续走代理
import os
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import sys
import json
import time
import shutil
import urllib.request
import urllib.error
import subprocess
import gc

# 只对本地 ComfyUI URL 禁用代理，其余地址继续使用系统代理
class _LocalBypassProxyHandler(urllib.request.ProxyHandler):
    def proxy_open(self, req, proxy, type):
        # 本地 ComfyUI 请求不走代理
        if req.host.startswith("127.0.0.1") or req.host.startswith("localhost"):
            return None
        return super().proxy_open(req, proxy, type)

urllib.request.install_opener(
    urllib.request.build_opener(_LocalBypassProxyHandler())
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    log, save_json, load_json, run_cmd,
    EPISODE_NUM, get_dirs, WAN22_MODELS_DIR,
    COMFYUI_GPU_FLAG, _IS_APPLE_SILICON, _IS_MAC, _IS_KAGGLE,
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
    for candidate in ["/kaggle/working/ComfyUI", "/kaggle/working/ComfyUI-master",
                      "/Users/heipi/ComfyUI"]:
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


def _create_extra_model_paths(comfyui_dir=None):
    """创建 extra_model_paths.yaml 注册模型路径"""
    if comfyui_dir is None:
        comfyui_dir = COMFYUI_DIR
    rel_base = "./models" if _IS_MAC else "."
    if _IS_MAC:
        yaml_content = f"""\
wan22_ti2v:
  base_path: {WAN22_MODELS_DIR}
  diffusion_models: ./unet
  text_encoders: ./text_encoders
  vae: ./vae
"""
    else:
        yaml_content = f"""\
wan22_ti2v:
  base_path: {WAN22_MODELS_DIR}
  diffusion_models: {rel_base}
  text_encoders: {rel_base}
  vae: {rel_base}
"""
    yaml_path = f"{comfyui_dir}/extra_model_paths.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    log(f"  extra_model_paths.yaml: {yaml_path}")


def _start_comfyui():
    """启动 ComfyUI 服务器"""
    # 绕过代理访问本地 ComfyUI（必须尽早设置，urllib 读取环境变量有缓存）
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    # 清除 urllib 的代理缓存，强制重新读取 no_proxy
    import urllib.request
    urllib.request.getproxies = lambda: {}
    # 先检测是否已有实例在运行（多次重试，处理503）
    for attempt in range(8):
        try:
            req = urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3)
            if req.status == 200:
                log(f"  ✅ ComfyUI 已在运行 ({COMFYUI_URL})")
                return True
        except urllib.error.HTTPError as e:
            if e.code in (503, 502, 409):
                if attempt == 0:
                    log(f"  ⏳ ComfyUI 响应 {e.code} (启动中)，继续等待...")
                time.sleep(3)
                continue
            log(f"  ⏳ ComfyUI 响应 {e.code}，继续等待...")
            time.sleep(3)
            continue
        except Exception:
            break
    # 额外检测：看端口 8188 是否监听
    import subprocess
    try:
        result = subprocess.run(
            ["lsof", "-iTCP:8188", "-sTCP:LISTEN", "-P", "-n"],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            log(f"  ✅ ComfyUI 端口 8188 正在监听，已在运行")
            return True
    except Exception:
        pass
    log(f"  ⚠️ ComfyUI 未运行，尝试启动...")

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

    comfyui_dir = result[1] if isinstance(result[1], str) and os.path.isdir(result[1]) else COMFYUI_DIR
    _create_extra_model_paths(comfyui_dir)

    cwd = comfyui_dir
    log_file = os.path.join(comfyui_dir, "comfyui.log")
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

    # Mac 本地: 模型在 models/unet/, models/vae/, models/clip/, models/text_encoders/
    if _IS_MAC:
        for sub in ["unet", "vae", "clip", "text_encoders"]:
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


def _wait_for_completion(prompt_id, timeout=3600):
    """等待工作流完成（不修改全局代理设置）"""
    import urllib.request as _urllib_req
    # 使用本地绕过代理的 opener（不污染全局 getproxies）
    class _LocalOpener(_urllib_req.OpenerDirector):
        def open(self, fullurl, data=None, timeout=30):
            if isinstance(fullurl, str) and ("127.0.0.1" in fullurl or "localhost" in fullurl):
                # 临时移除代理
                saved = _urllib_req.getproxies
                _urllib_req.getproxies = lambda: {}
                try:
                    return super().open(fullurl, data, timeout)
                finally:
                    _urllib_req.getproxies = saved
            return super().open(fullurl, data, timeout)

    opener = _LocalOpener()
    opener.add_handler(_urllib_req.ProxyHandler({}))
    # 移除默认的 ProxyHandler，只对本地 URL 绕过
    _urllib_req.install_opener(_urllib_req.build_opener(
        _LocalBypassProxyHandler(),
        _urllib_req.HTTPHandler(),
        _urllib_req.HTTPSHandler(),
        _urllib_req.HTTPDefaultErrorHandler(),
        _urllib_req.HTTPRedirectHandler(),
    ))

    start = time.time()
    last_log = 0
    while time.time() - start < timeout:
        try:
            req = _urllib_req.Request(f"{COMFYUI_URL}/history/{prompt_id}")
            with _urllib_req.urlopen(req, timeout=60) as resp:
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
    """生成占位视频（纯 ffmpeg，不依赖 PIL）"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dur = max(num_frames / 8, 1)
    # 用 lavfi color source 生成纯色占位视频
    cmd = (
        f'ffmpeg -y -f lavfi -i "color=c=1a1a2e:s=832x480:d={dur}:r=8" '
        f'-c:v libx264 -pix_fmt yuv420p -movflags +faststart '
        f'"{output_path}" 2>/dev/null'
    )
    subprocess.run(cmd, shell=True, timeout=30)


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

    # 参数 — 从环境变量读取模型配置（由 kaggle_pipeline 传入）
    model_name = os.environ.get("WAN22_MODEL", "wan2.2-5b-f16")
    _fps = int(os.environ.get("WAN22_FPS", "8"))
    steps = int(os.environ.get("WAN22_STEPS", "20"))
    cfg = float(os.environ.get("WAN22_CFG", "5.0"))
    shift = float(os.environ.get("WAN22_SHIFT", "8.0"))

    # 尺寸映射
    SIZE_MAP = {1: (384, 384), 2: (834, 480), 3: (480, 834), 4: (576, 320), 5: (384, 640)}
    size_idx = int(os.environ.get("WAN22_SIZE", "2"))
    w, h = SIZE_MAP.get(size_idx, (834, 480))

    sampler = "euler"
    scheduler = "simple"

    # 动态帧数：根据每个镜头 duration_seconds 计算
    _default_num_frames = 6 * _fps

    style_name = storyboard.get("style", "未指定")
    log(f"模型: {model_name} | {w}x{h} | {_default_num_frames}f (默认) | {steps}步 | CFG={cfg} | shift={shift} | 风格: {style_name}")

    count = 0
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            ep = storyboard.get("episode", 1)
            out = f"{dirs['videos']}/ep{ep:02d}_{scene['scene_id']}_{sid}.mp4"

            # --force 模式下不跳过（主流程已清理了 episode 目录）
            # 检查是否为有效视频（>500KB 视为正常生成，<100KB 视为占位/损坏）
            if os.path.exists(out) and 100000 < os.path.getsize(out) < 5000000:
                log(f"  [{count}/{total}] {sid} 跳过 ({os.path.getsize(out)//1024}KB)")
                continue
            elif os.path.exists(out) and os.path.getsize(out) <= 100000:
                log(f"  [{count}/{total}] {sid} 覆盖占位视频 ({os.path.getsize(out)//1024}KB)")
                os.remove(out)

            # ComfyUI 由 kaggle_pipeline 统一启动（SWAP > 3G 才重启）

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

            # 动态帧数：根据镜头 duration_seconds 计算
            duration_seconds = shot.get("duration_seconds", 6)
            num_frames = max(int(duration_seconds * _fps), 9)

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
                    # 视频复制成功 → kill ComfyUI 释放内存/SWP
                    _kill_comfyui()
                    log(f"    ⏹ ComfyUI 已停止 (释放内存)")
                else:
                    log(f"  [{count}/{total}] {sid} 无输出")
                    _save_placeholder_video(shot, out, shot.get("duration_seconds", 6) * _fps)

            except Exception as e:
                log(f"  [{count}/{total}] {sid} 失败: {e}")
                # 超时后检查 ComfyUI 是否已生成视频（可能轮询太慢）
                video_output = _check_existing_output(completion, prompt_id) if 'completion' in dir() else None
                if not video_output:
                    # 检查 ComfyUI output 目录是否有最新文件
                    video_output = _poll_comfyui_output()
                if video_output and os.path.isfile(video_output):
                    shutil.copy2(video_output, out)
                    size_mb = os.path.getsize(out) / 1e6
                    log(f"  [{count}/{total}] {sid} ✓ 超时后找到 ({size_mb:.1f}MB)")
                    _kill_comfyui()
                    log(f"    ⏹ ComfyUI 已停止 (释放内存)")
                else:
                    _save_placeholder_video(shot, out, shot.get("duration_seconds", 6) * _fps)
                    log(f"  [{count}/{total}] {sid} 使用占位视频")

    log("视频生成完成")


def _kill_comfyui():
    """Kill ComfyUI 进程释放内存/SWP"""
    try:
        subprocess.run(
            'pkill -f "main.py.*ComfyUI" 2>/dev/null; pkill -f "python.*main.py" 2>/dev/null',
            shell=True, timeout=10
        )
        # 强制清理残留
        subprocess.run(
            'pkill -9 -f "main.py" 2>/dev/null',
            shell=True, timeout=5
        )
        log("    🔄 ComfyUI 进程已清理，等待 SWAP 释放...")
        time.sleep(5)  # 给系统时间释放内存
    except Exception as e:
        log(f"    ⚠️ 清理 ComfyUI 失败: {e}")


def _restart_comfyui():
    """调用桌面 WAN2.2-F16.command 重启 ComfyUI（kill 旧进程 → 释放内存 → 启动新实例）"""
    # Kill 旧进程
    _kill_comfyui()
    # 启动 ComfyUI（通过桌面 .command 文件，会在独立终端窗口中运行）
    log("    🚀 启动 ComfyUI (WAN2.2-F16.command)...")
    try:
        subprocess.Popen(
            ['open', '-a', 'Terminal', '/Users/heipi/Desktop/WAN2.2-F16.command'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # 给 .command 文件时间执行 pkill + 启动
        time.sleep(5)
    except Exception as e:
        log(f"    ⚠️ 启动 ComfyUI 失败: {e}")


def _wait_for_comfyui(timeout=300):
    """等待 ComfyUI 真正就绪（连续3次成功提交空请求）"""
    import urllib.error
    start = time.time()
    success_count = 0
    while time.time() - start < timeout:
        try:
            test_payload = json.dumps({"prompt": {}}).encode()
            req = urllib.request.Request(
                f"{COMFYUI_URL}/prompt",
                data=test_payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                success_count += 1
            # 连续3次成功才认为真正就绪（避免端口监听但队列未就绪）
            if success_count >= 3:
                return True
        except urllib.error.HTTPError:
            success_count += 1
            if success_count >= 3:
                return True
        except:
            success_count = 0  # 重置计数
        time.sleep(3)
    return False


def _get_swap_used_gb():
    """获取当前 swap 使用量（GB）"""
    try:
        result = subprocess.run(
            ['sysctl', 'vm.swapusage'],
            capture_output=True, text=True, timeout=5
        )
        # 输出格式: vm.swapusage: total = 3072.00M  used = 1829.50M  free = 1242.50M  (encrypted)
        output = result.stdout
        import re
        match = re.search(r'used\s*=\s*([\d.]+)M', output)
        if match:
            return float(match.group(1)) / 1024  # MB → GB
    except:
        pass
    return 0


def _purge_swap():
    """主动执行 sudo purge 降低 swap"""
    try:
        result = subprocess.run(
            ['bash', '-c', 'printf "55709521pingguo\n" | sudo -S purge'],
            capture_output=True, text=True, timeout=30, shell=True
        )
        return result.returncode == 0
    except:
        return False


def _wait_for_swap_release(limit_gb=3, timeout=300):
    """等待 swap 使用量降到 limit_gb GB 以下
    
    策略: 至少等待 30s，期间每 30s 主动 purge 一次。
    只有当 swap < limit_gb 才继续，超时则最后再 purge 一次。
    """
    start = time.time()
    last_purge_time = 0
    min_wait = 30  # 至少等待 30 秒

    # 第一阶段: 至少等 min_wait 秒，期间每 30 秒 purge
    while time.time() - start < min_wait:
        time.sleep(5)
        elapsed = time.time() - start
        # 每 30 秒主动 purge（包括 0s 时先 purge 一次）
        if elapsed - last_purge_time >= 30 or last_purge_time == 0:
            log(f"    🔧 主动 purge (已等 {elapsed:.0f}s)...")
            if _purge_swap():
                log(f"    🔧 purge 完成")
            else:
                log(f"    ⚠️ purge 失败或超时")
            last_purge_time = elapsed
            time.sleep(3)  # 等 purge 生效

    # 第二阶段: 检查 swap 是否安全
    swap_gb = _get_swap_used_gb()
    if swap_gb < limit_gb:
        log(f"    ✅ SWAP {swap_gb:.1f}GB < {limit_gb}GB — 安全 (等待了 {time.time()-start:.0f}s)")
        return True

    # 第三阶段: 继续等待，每 30 秒再 purge，直到 timeout
    log(f"    ⏳ SWAP {swap_gb:.1f}GB 仍高于 {limit_gb}GB，继续降压...")
    while time.time() - start < timeout:
        time.sleep(5)
        elapsed = time.time() - start
        swap_gb = _get_swap_used_gb()
        if swap_gb < limit_gb:
            log(f"    ✅ SWAP {swap_gb:.1f}GB < {limit_gb}GB — 安全 (等待了 {elapsed:.0f}s)")
            return True
        # 每 30 秒再 purge 一次
        if elapsed - last_purge_time >= 30:
            log(f"    🔧 主动 purge (已等 {elapsed:.0f}s)...")
            if _purge_swap():
                log(f"    🔧 purge 完成")
            last_purge_time = elapsed
            time.sleep(3)

    # 超时: 最后再 purge 一次
    log(f"    ⚠️ SWAP 等待超时 ({timeout}s)，最后强制 purge...")
    _purge_swap()
    time.sleep(5)
    swap_gb = _get_swap_used_gb()
    if swap_gb < limit_gb:
        log(f"    ✅ purge 后 SWAP {swap_gb:.1f}GB — 安全")
        return True

    log(f"    ⚠️ 无法降低 SWAP (当前 {swap_gb:.1f}GB)，继续（可能 thrashing）")
    return False

def _check_existing_output(completion, prompt_id):
    """从已完成的 completion 数据中找视频输出"""
    if not completion:
        return None
    outputs = completion.get("outputs", {})
    for nid, nout in outputs.items():
        if "gifs" in nout:
            gifs = nout["gifs"]
            if isinstance(gifs, list) and gifs:
                video_output = gifs[0]
                if isinstance(video_output, dict):
                    video_output = video_output.get("fullpath", "")
                if video_output and os.path.isfile(video_output):
                    return video_output
    return None


def _poll_comfyui_output():
    """轮询 ComfyUI output 目录找最新视频"""
    import glob as _glob
    out_dir = os.path.expanduser("~/ComfyUI/output")
    if not os.path.isdir(out_dir):
        # 尝试其他常见路径
        for alt in ["/Users/heipi/ComfyUI/output"]:
            if os.path.isdir(alt):
                out_dir = alt
                break
    if not os.path.isdir(out_dir):
        return None
    files = _glob.glob(os.path.join(out_dir, "wan22_*.mp4"))
    if files:
        # 取最新的
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
    return None


if __name__ == "__main__":
    dirs = get_dirs(EPISODE_NUM)
    storyboard = load_json(f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json")
    main(storyboard)
