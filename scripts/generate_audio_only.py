#!/usr/bin/env python3
"""
独立音频生成脚本 - 根据已有视频片段生成配音
用法: python generate_audio_only.py --storyboard path/to/storyboard.json --output-dir path/to/audio/
"""
import sys, os, json, argparse, subprocess, time, asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_json, log, AUDIO_SAMPLE_RATE, VOICE_PARAMS, EMOTION_SPEED

# 检查 TTS 可用性
def check_tts():
    chat = None
    edge_ok = False
    # 跳过 ChatTTS（首次使用需要下载大模型，走 edge-tts 即可）
    # try:
    #     import ChatTTS
    #     chat = ChatTTS.Chat()
    #     chat.load(compile=False)
    #     log("ChatTTS ✓")
    # except Exception as e:
    #     log(f"ChatTTS: {e}")
    try:
        import edge_tts
        edge_ok = True
        log("edge-tts ✓")
    except:
        pass
    return chat, edge_ok

def get_video_duration(video_path):
    """获取视频时长（秒）"""
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=5
        )
        if r.stdout.strip():
            return float(r.stdout.strip())
    except:
        pass
    return 0

def generate_edge_tts(text, voice, rate, out_path):
    """用 edge-tts 生成音频"""
    import edge_tts
    async def _gen():
        c = edge_tts.Communicate(text, voice, rate=rate)
        await c.save(out_path)
    asyncio.run(_gen())

def save_silence(out_path, duration):
    """生成静音 wav"""
    import struct, wave
    sr = AUDIO_SAMPLE_RATE
    n_frames = int(sr * duration)
    with wave.open(out_path, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b'\x00\x00' * n_frames)

def truncate_audio(audio_path, max_duration):
    """将音频截断到指定时长"""
    tmp = audio_path + '.tmp.wav'
    subprocess.run([
        'ffmpeg', '-y', '-i', audio_path,
        '-t', str(max_duration),
        '-acodec', 'copy',
        tmp
    ], capture_output=True, timeout=30)
    if os.path.exists(tmp):
        os.replace(tmp, audio_path)

def main():
    parser = argparse.ArgumentParser(description="独立音频生成（根据视频片段）")
    parser.add_argument("--storyboard", required=True, help="storyboard.json 路径")
    parser.add_argument("--output-dir", required=True, help="音频输出目录")
    parser.add_argument("--videos-dir", help="视频文件目录（默认从 storyboard 推断）")
    parser.add_argument("--regenerate", action="store_true", help="重新生成已有音频")
    args = parser.parse_args()

    sb = load_json(args.storyboard)
    episode = sb.get("episode", 1)
    os.makedirs(args.output_dir, exist_ok=True)

    # 确定视频目录
    videos_dir = args.videos_dir
    if not videos_dir:
        ep_dir = os.path.dirname(os.path.dirname(args.storyboard))
        videos_dir = os.path.join(ep_dir, "videos")

    log(f"集数: episode_{episode:02d}")
    log(f"视频目录: {videos_dir}")
    log(f"输出目录: {args.output_dir}")

    chat, edge_ok = check_tts()
    if not chat and not edge_ok:
        log("❌ 无可用 TTS 引擎")
        return

    # 角色 → edge-tts 声音映射
    EDGE_V = {
        "xiaoming": "zh-CN-YunxiNeural",
        "xiaoli": "zh-CN-XiaoxiaoNeural",
        "boss_wang": "zh-CN-YunjianNeural",
        "narrator": "zh-CN-YunxiNeural",
        "小丽": "zh-CN-XiaoxiaoNeural",
        "王总": "zh-CN-YunjianNeural",
        "小明": "zh-CN-YunxiNeural",
    }

    total = sum(len(s.get("shots", [])) for s in sb.get("scenes", []))
    count = 0

    for scene in sb.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            out = f"{args.output_dir}/ep{episode:02d}_{scene['scene_id']}_{sid}.wav"

            # 跳过已有文件（除非 --regenerate）
            if os.path.exists(out) and not args.regenerate:
                log(f"  [{count}/{total}] {sid} 跳过（已存在）")
                continue

            # 获取实际视频时长
            video_path = f"{videos_dir}/ep{episode:02d}_{scene['scene_id']}_{sid}.mp4"
            video_dur = get_video_duration(video_path) if os.path.exists(video_path) else float(shot.get("duration_seconds", 3))

            # 优先 dialogue，没有则 narration
            char = shot.get("character", "narrator")
            text = shot.get("dialogue") or shot.get("narration") or ""
            emotion = shot.get("emotion", "calm")

            if not text:
                save_silence(out, video_dur)
                log(f"  [{count}/{total}] {sid} 静音（无文本）")
                continue

            # 计算目标语速（中文约4字/秒，edge-tts rate 百分比）
            target_dur = video_dur
            chars = len(text)
            ideal_rate = chars / target_dur / 4.0  # 正常语速4字/s的倍率
            base_rate = f"{int((ideal_rate - 1) * 100):+d}%"

            # 限制 rate 在合理范围
            rate_pct = int((ideal_rate - 1) * 100)
            if rate_pct > 50:
                base_rate = "+50%"
            elif rate_pct < -30:
                base_rate = "-30%"

            log(f"  [{count}/{total}] {sid} ({char}) {chars}字 {video_dur:.1f}s rate={base_rate}")

            ok = False

            # 尝试 ChatTTS
            if chat is not None:
                try:
                    wavs = chat.infer([text])
                    if wavs and len(wavs) > 0:
                        import torchaudio
                        audio = wavs[0]
                        if isinstance(audio, torch.Tensor):
                            torchaudio.save(out, audio.unsqueeze(0), AUDIO_SAMPLE_RATE)
                            ok = True
                            log(f"    ✅ ChatTTS 生成")
                except Exception as e:
                    log(f"    ChatTTS 失败: {e}")

            # fallback edge-tts
            if not ok and edge_ok:
                try:
                    voice = EDGE_V.get(char, "zh-CN-YunxiNeural")
                    generate_edge_tts(text, voice, base_rate, out)
                    ok = True
                    log(f"    ✅ edge-tts 生成")
                except Exception as e:
                    log(f"    edge-tts 失败: {e}")

            # 检查时长，截断或补静音
            if ok and os.path.exists(out):
                actual_dur = get_video_duration(out)  # wav 也可以用 ffprobe
                if actual_dur > video_dur + 0.1:
                    truncate_audio(out, video_dur)
                    log(f"    ✂️ 截断 {actual_dur:.1f}s → {video_dur:.1f}s")
                elif actual_dur < video_dur - 0.5:
                    # 音频太短，补静音尾部
                    tmp = out + '.tmp.wav'
                    subprocess.run([
                        'ffmpeg', '-y', '-i', out, '-f', 'lavfi', '-i', f'anullsrc=cl=mono:sr={AUDIO_SAMPLE_RATE}',
                        '-filter_complex', f'[0:a][1:a]concat=n=2:v=0:a=1[out]',
                        '-map', '[out]', '-t', str(video_dur),
                        tmp
                    ], capture_output=True, timeout=30)
                    if os.path.exists(tmp):
                        os.replace(tmp, out)
                        log(f"    🔧 补尾 {actual_dur:.1f}s → {video_dur:.1f}s")

            if not ok:
                save_silence(out, video_dur)
                log(f"    ⚠️ 使用静音")

    log(f"\n完成! 音频输出: {args.output_dir}")

if __name__ == "__main__":
    main()
