#!/usr/bin/env python3
"""Step 5: 配音生成 - ChatTTS / edge-tts"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="AI短剧配音生成")
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--output-dir", default="output/audio")
    args = parser.parse_args()

    sb = load_json(args.storyboard)
    dirs = get_dirs(sb.get("episode", 1))
    total = sum(len(s.get("shots", [])) for s in sb.get("scenes", []))

    # ChatTTS
    chat = None
    try:
        import ChatTTS
        chat = ChatTTS.Chat()
        chat.load(compile=False)
        log("ChatTTS ✓")
    except Exception as e:
        log(f"ChatTTS: {e}")

    # edge-tts 备用
    edge_ok = False
    try:
        import edge_tts
        edge_ok = True
    except:
        pass

    EDGE_V = {
        "xiaoming": "zh-CN-YunxiNeural",
        "xiaoli": "zh-CN-XiaoxiaoNeural",
        "boss_wang": "zh-CN-YunjianNeural",
        "narrator": "zh-CN-YunxiNeural"
    }

    count = 0
    for scene in sb.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            ep = sb.get("episode", 1)
            out = f"{args.output_dir}/ep{ep:02d}_{scene['scene_id']}_{sid}.wav"
            os.makedirs(args.output_dir, exist_ok=True)

            if os.path.exists(out):
                continue

            char = shot.get("character", "narrator")
            # 优先用 dialogue，没有则用 narration
            text = shot.get("dialogue") or shot.get("narration") or ""
            emotion = shot.get("emotion", "calm")
            dur = shot.get("duration_seconds", 3)

            if not text:
                _save_silence(out, float(dur))
                continue

            # 获取实际视频时长，优先使用视频真实时长
            video_dur = float(dur)
            video_path = f"{dirs['videos']}/ep{ep:02d}_{scene['scene_id']}_{sid}.mp4"
            if os.path.exists(video_path):
                try:
                    r = subprocess.run(
                        ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path],
                        capture_output=True, text=True, timeout=5
                    )
                    if r.stdout.strip():
                        video_dur = float(r.stdout.strip())
                except:
                    pass

            voice = VOICE_PARAMS.get(char, VOICE_PARAMS["narrator"]).copy()
            base_speed = voice.get("speed", 1.0) * EMOTION_SPEED.get(emotion, 1.0)
            # edge-tts >=7.0 需要百分比格式 rate (如 "+10%", "-15%")
            base_rate = f"{int((base_speed - 1) * 100):+d}%"

            # 先用默认语速生成，再根据实际时长调整语速重新生成
            ok = False
            generated_audio = None

            # ChatTTS
            if chat is not None:
                try:
                    wavs = chat.infer([text])
                    if wavs and len(wavs) > 0:
                        import torchaudio
                        audio = wavs[0]
                        if isinstance(audio, torch.Tensor):
                            # 先保存，后续检查时长
                            os.makedirs(os.path.dirname(out), exist_ok=True)
                            torchaudio.save(out, audio.unsqueeze(0), AUDIO_SAMPLE_RATE)
                            generated_audio = audio
                            ok = True
                except:
                    pass

            # edge-tts
            if not ok and edge_ok:
                try:
                    import asyncio
                    vn = EDGE_V.get(char, "zh-CN-YunxiNeural")

                    async def _t():
                        c = edge_tts.Communicate(text, vn, rate=base_rate)
                        await c.save(out)

                    asyncio.run(_t())
                    ok = True
                except:
                    pass

            # 检查生成时长，如果超过视频时长则调整语速
            if ok and os.path.exists(out):
                try:
                    r = subprocess.run(
                        ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', out],
                        capture_output=True, text=True, timeout=5
                    )
                    actual_dur = float(r.stdout.strip())
                    # 如果音频超过视频时长 0.3s 以上，需要调整
                    if actual_dur > video_dur + 0.3:
                        if generated_audio is not None:
                            # ChatTTS: 不支持 rate 参数，用 ffmpeg 调整速度
                            speed_factor = actual_dur / video_dur
                            # atempo 范围 0.5-2.0，如果超出则分段处理
                            if speed_factor <= 2.0:
                                atempo = speed_factor
                            else:
                                atempo = 2.0  # 最大加速 2x
                            log(f"    ⚡ 音频{actual_dur:.1f}s > 视频{video_dur:.1f}s，ffmpeg 加速 {atempo:.2f}x")
                            tmp = out + ".tmp.wav"
                            subprocess.run(
                                ['ffmpeg', '-y', '-i', out, '-filter:a', f'atempo={atempo}',
                                 '-to', str(video_dur), tmp],
                                capture_output=True, timeout=30
                            )
                            os.replace(tmp, out)
                            # 如果加速后还超长（atempo 限制），截断
                            if actual_dur > video_dur * atempo:
                                subprocess.run(
                                    ['ffmpeg', '-y', '-i', out, '-t', str(video_dur),
                                     '-acodec', 'copy', tmp],
                                    capture_output=True, timeout=10
                                )
                                os.replace(tmp, out)
                        else:
                            # edge-tts: 加速文本重新生成
                            speed_factor = actual_dur / video_dur
                            new_speed = base_speed * speed_factor
                            rate_str = f"+{int((new_speed - 1) * 100)}%"
                            log(f"    ⚡ 音频{actual_dur:.1f}s > 视频{video_dur:.1f}s，加速 {rate_str} 重新生成")
                            import asyncio
                            vn = EDGE_V.get(char, "zh-CN-YunxiNeural")

                            async def _t2():
                                c = edge_tts.Communicate(text, vn, rate=rate_str)
                                await c.save(out)

                            asyncio.run(_t2())
                except:
                    pass

            # 最终兜底：音频绝对不能超过视频时长
            if ok and os.path.exists(out):
                try:
                    r = subprocess.run(
                        ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', out],
                        capture_output=True, text=True, timeout=5
                    )
                    actual_dur = float(r.stdout.strip())
                    if actual_dur > video_dur + 0.1:
                        tmp = out + ".tmp.wav"
                        subprocess.run(
                            ['ffmpeg', '-y', '-i', out, '-t', str(video_dur), '-acodec', 'copy', tmp],
                            capture_output=True, timeout=10
                        )
                        os.replace(tmp, out)
                        log(f"    ✂️ 截断到 {video_dur}s")
                except:
                    pass

            if not ok:
                _save_silence(out, video_dur)

            log(f"  [{count}/{total}] {sid} ({char}) {'✓' if ok else '静音'}")

    log("配音生成完成")


def _save_silence(output_path, duration):
    """生成静音 WAV 文件"""
    try:
        import wave, struct
        sr = AUDIO_SAMPLE_RATE
        n = int(sr * duration)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with wave.open(output_path, 'w') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            for _ in range(n): w.writeframes(struct.pack('<h', 0))
    except Exception:
        subprocess.run(
            f'ffmpeg -y -f lavfi -i "anullsrc=r={AUDIO_SAMPLE_RATE}:cl=mono" '
            f'-t {duration} -acodec pcm_s16le "{output_path}" 2>/dev/null',
            shell=True, timeout=30,
        )


if __name__ == "__main__":
    main()
