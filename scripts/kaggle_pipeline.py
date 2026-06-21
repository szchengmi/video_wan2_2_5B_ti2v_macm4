#!/usr/bin/env python3
"""
Wan2.2 TI2V 5B AI短剧端到端流水线
===================================
在Kaggle Notebook中运行此脚本，自动完成：
  1. 剧本生成 (Gemini API / 本地 Qwen)
  2. 分镜生成 (结构化JSON)
  3. 视频生成 (Wan2.2 TI2V 5B via ComfyUI，跳过 SD 1.5)
  4. 配音生成 (ChatTTS / edge-tts)
  5. 剪辑合成 (FFmpeg)

与 kaggle-ai-series 的区别：
  - 跳过 Step 3 (SD 1.5 画面生成)
  - Step 4 使用 Wan2.2 TI2V 5B 直接文本生成视频
  - 模型路径: /kaggle/input/saysnkaggle/wan2-2-5b-f16/

Kaggle 运行:
  !rm -rf /kaggle/working/* && cd /kaggle/working && git clone https://github.com/szchengmi/video_wan2_2_5B_ti2v.git && cd video_wan2_2_5B_ti2v/scripts && python kaggle_pipeline.py --force
"""

import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

# 导入 common 中的配置和工具
from common import (
    BASE_DIR, EPISODE_NUM, WAN22_DATASET, WAN22_MODELS_DIR,
    get_dirs, setup_dirs, log, run_cmd, save_json, load_json,
    _IS_MAC, _IS_APPLE_SILICON, _IS_KAGGLE,
)

import time
import shutil
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V AI短剧生成")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--duration", type=int, default=15, help="目标时长(秒)")
    parser.add_argument("--style", type=int, default=1, help="风格: 1=二次元 2=古代田园 3=赛博朋克 4=动漫 5=类真人 6=火柴人")
    parser.add_argument("--model", type=int, default=1, help="模型选择: 1=Wan2.2-5B-F16 2=Wan2.2-5B-GGUF 3=Wan2.1-1.3B-F16 4=Wan2.1-14B-GGUF")
    parser.add_argument("--size", type=int, default=2, help="尺寸: 1=384×384 2=834×480 3=480×834 4=576×320 5=384×640")
    parser.add_argument("--step", type=int, default=None, help="指定单步运行: 1=剧本 2=分镜 4=视频 5=配音 6=合成 (不指定=全部运行)")
    args = parser.parse_args()

    # 模型选择映射
    MODEL_MAP = {
        1: {"name": "wan2.2-5b-f16", "model_arg": "wan2.2-5b-f16", "fps": 8, "steps": 20, "cfg": 5.0, "shift": 8.0},
        2: {"name": "wan2.2-5b-gguf", "model_arg": "wan2.2-5b-gguf", "fps": 8, "steps": 20, "cfg": 5.0, "shift": 8.0},
        3: {"name": "wan2.1-1.3b-f16", "model_arg": "wan2.1-1.3b-f16", "fps": 16, "steps": 30, "cfg": 6.0, "shift": 4.0},
        4: {"name": "wan2.1-14b-gguf", "model_arg": "wan2.1-14b-gguf", "fps": 16, "steps": 30, "cfg": 6.0, "shift": 4.0},
    }
    model_cfg = MODEL_MAP.get(args.model, MODEL_MAP[1])

    # 风格选择映射
    STYLE_MAP = {1: "二次元", 2: "古代田园", 3: "赛博朋克", 4: "动漫", 5: "类真人", 6: "火柴人"}
    style_name = STYLE_MAP.get(args.style, "二次元")

    if args.episode is not None:
        global EPISODE_NUM
        EPISODE_NUM = args.episode

    log("╔══════════════════════════════════════════╗")
    log("║   AI短剧 — Wan2.2 TI2V 5B Pipeline       ║")
    log("╚══════════════════════════════════════════╝")
    log(f"集数: {EPISODE_NUM} | 模型: {WAN22_MODELS_DIR}")

    # 列出所有可用选项
    MODEL_NAMES = {1: "Wan2.2-5B-F16", 2: "Wan2.2-5B-GGUF", 3: "Wan2.1-1.3B-F16", 4: "Wan2.1-14B-GGUF"}
    SIZE_NAMES = {1: "384×384", 2: "834×480", 3: "480×834", 4: "576×320", 5: "384×640"}
    log(f"  模型: {args.model}={MODEL_NAMES.get(args.model, '?')} | 可用: 1=Wan2.2-5B-F16 2=Wan2.2-5B-GGUF 3=Wan2.1-1.3B-F16 4=Wan2.1-14B-GGUF")
    log(f"  尺寸: {args.size}={SIZE_NAMES.get(args.size, '?')} | 可用: 1=384×384 2=834×480 3=480×834 4=576×320 5=384×640")
    log(f"  风格: {args.style}={style_name} | 可用: 1=二次元 2=古代田园 3=赛博朋克 4=动漫 5=类真人 6=火柴人")
    log(f"  时长: {args.duration}秒")
    if args.step:
        log(f"  单步模式: 只运行 Step {args.step}")
    log("")

    # 清除旧输出（仅全量运行或 Step1 时清理，单步模式不删）
    if args.force and args.step is None or args.step == 1:
        log("⚠️ 强制重新生成")
        ep_dir = get_dirs(EPISODE_NUM)["episode"]
        if os.path.isdir(ep_dir):
            shutil.rmtree(ep_dir)
            log(f"  已清除: {ep_dir}")

    setup_dirs()

    # 检查并下载模型 (Mac 本地模式下)
    if not _IS_KAGGLE:
        models_needed = [
            f"{WAN22_MODELS_DIR}/unet/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
            f"{WAN22_MODELS_DIR}/vae/split_files/vae/wan2.2_vae.safetensors",
            f"{WAN22_MODELS_DIR}/clip/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        ]
        missing = [m for m in models_needed if not os.path.exists(m)]
        if missing:
            log(f"\n⚠️ 缺少 {len(missing)} 个模型文件，开始下载...")
            subprocess.run(["python", "download_models.py"], cwd=_SCRIPT_DIR, check=True)
        else:
            log("  ✅ 模型文件已存在")

    # 安装依赖
    log("\n安装依赖...")
    if _IS_APPLE_SILICON:
        log("  MPS 模式: 确保 torch >= 2.0 已安装")
    subprocess.run("pip install -q edge-tts psutil", shell=True, timeout=60)

    t0 = time.time()

    # ══════════════════════════════════════════
    # Step 1: 剧本生成
    # ══════════════════════════════════════════
    if args.step is None or args.step == 1:
        log("\n" + "=" * 50)
        log("Step 1: 剧本生成")
        log("=" * 50)
        from step1_generate_story import generate_script
        script = generate_script(EPISODE_NUM, duration_minutes=args.duration, style=style_name)
        log(f"剧本: {script.get('title')} | 风格: {script.get('style', style_name)}")

        # 保存剧本
        script_path = f"{get_dirs(EPISODE_NUM)['storyboard']}/episode_{EPISODE_NUM:02d}_script.json"
        save_json(script, script_path)

    # ══════════════════════════════════════════
    # Step 2: 分镜生成
    # ══════════════════════════════════════════
    if args.step is None or args.step == 2:
        log("\n" + "=" * 50)
        log("Step 2: 分镜生成")
        log("=" * 50)

        # 加载已有剧本（单步模式需要）
        if args.step == 2:
            script_path = f"{get_dirs(EPISODE_NUM)['storyboard']}/episode_{EPISODE_NUM:02d}_script.json"
            script = load_json(script_path)
        from step2_generate_storyboard import generate_storyboard
        storyboard = generate_storyboard(script)
        total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))
        log(f"分镜: {len(storyboard['scenes'])}场景 | {total}镜头")

        # 保存 storyboard
        sb_path = f"{get_dirs(EPISODE_NUM)['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json"
        save_json(storyboard, sb_path)
    else:
        # 非 step2 但后续步骤需要 storyboard 路径
        sb_path = f"{get_dirs(EPISODE_NUM)['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json"

    # Step 3: 跳过 (Wan2.2 直接生成视频，不需要先画图)
    if args.step is None:
        log("\n" + "=" * 50)
        log("Step 3: 画面生成 → 跳过 (Wan2.2 直接 T2V)")
        log("=" * 50)

    # Step 4: 视频生成 (Wan2.2 TI2V)
    if args.step is None or args.step == 4:
        # Step 4~5: 检查 ComfyUI 内存，SWAP > 3G 才重启
        import psutil
        comfy_ps = None
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if p.info['cmdline'] and 'main.py' in ' '.join(p.info['cmdline']):
                    comfy_ps = p
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if comfy_ps:
            swap_info = psutil.swap_memory()
            swap_used_gb = swap_info.used / (1024**3)
            if swap_used_gb > 3:
                log(f"⚠️ SWAP={swap_used_gb:.1f}GB > 3G，重启 ComfyUI 释放内存...")
                comfy_ps.kill()
                import time as _time
                _time.sleep(5)
                subprocess.Popen(
                    ["python", "main.py", "--fp16-unet", "--preview-method", "taesd"],
                    cwd="/Users/heipi/ComfyUI",
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                _time.sleep(120)
                log("  ComfyUI 重启完成")
            else:
                log(f"SWAP={swap_used_gb:.1f}GB ≤ 3G，无需重启")
        else:
            log("⚠️ 未找到 ComfyUI 进程")

        log("\n" + "=" * 50)
        log("Step 4: 视频生成 (Wan2.2 TI2V 5B)")
        log("=" * 50)
        videos_dir = f"{get_dirs(EPISODE_NUM)['videos']}"
        # 传递模型配置给 step4
        env = os.environ.copy()
        env["WAN22_MODEL"] = model_cfg["model_arg"]
        env["WAN22_FPS"] = str(model_cfg["fps"])
        env["WAN22_STEPS"] = str(model_cfg["steps"])
        env["WAN22_CFG"] = str(model_cfg["cfg"])
        env["WAN22_SHIFT"] = str(model_cfg["shift"])
        env["WAN22_SIZE"] = str(args.size)
        subprocess.run([
            sys.executable, "step4_generate_videos_wan22.py",
            "--storyboard", sb_path,
            "--output-dir", videos_dir,
        ], cwd=_SCRIPT_DIR, env=env)

    # Step 5: 配音生成 (在视频之后，根据实际视频时长调整语速)
    if args.step is None or args.step == 5:
        log("\n" + "=" * 50)
        log("Step 5: 配音生成")
        log("=" * 50)

        # 单步模式：重新加载 storyboard（step5 可能回写 duration_seconds）
        if args.step == 5:
            sb_path = f"{get_dirs(EPISODE_NUM)['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json"

        audio_dir = f"{get_dirs(EPISODE_NUM)['audio']}"
        subprocess.run([
            sys.executable, "step5_generate_audio.py",
            "--storyboard", sb_path,
            "--output-dir", audio_dir,
        ], cwd=_SCRIPT_DIR)

    # Step 6: 剪辑合成
    if args.step is None or args.step == 6:
        log("\n" + "=" * 50)
        log("Step 6: 剪辑合成")
        log("=" * 50)

        # 单步模式：检查视频和音频目录是否有文件
        videos_dir = f"{get_dirs(EPISODE_NUM)['videos']}"
        audio_dir = f"{get_dirs(EPISODE_NUM)['audio']}"
        final_dir = f"{get_dirs(EPISODE_NUM)['final']}"
        subprocess.run([
            sys.executable, "step6_compose.py",
            "--storyboard", sb_path,
            "--videos-dir", videos_dir,
            "--audio-dir", audio_dir,
            "--output-dir", final_dir,
        ], cwd=_SCRIPT_DIR)

    # 总结
    elapsed = (time.time() - t0) / 60
    log(f"\n{'=' * 50}")
    log(f"全部完成! 耗时: {elapsed:.1f} 分钟")
    if args.step:
        log(f"单步模式: 仅 Step {args.step}")
    log(f"{'=' * 50}")


if __name__ == "__main__":
    main()
