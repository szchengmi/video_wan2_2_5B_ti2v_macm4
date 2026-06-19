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
            text = shot.get("dialogue") or shot.get("narration") or ""
            emotion = shot.get("emotion", "calm")
            dur = shot.get("duration_seconds", 3)

            if not text:
                _save_silence(out, float(dur))
                continue

            voice = VOICE_PARAMS.get(char, VOICE_PARAMS["narrator"]).copy()
            voice["speed"] = voice.get("speed", 1.0) * EMOTION_SPEED.get(emotion, 1.0)

            ok = False

            # ChatTTS
            if chat is not None:
                try:
                    wavs = chat.infer([text])
                    if wavs and len(wavs) > 0:
                        import torchaudio
                        audio = wavs[0]
                        if isinstance(audio, torch.Tensor):
                            torchaudio.save(out, audio.unsqueeze(0), AUDIO_SAMPLE_RATE)
                        ok = True
                except:
                    pass

            # edge-tts
            if not ok and edge_ok:
                try:
                    import asyncio
                    vn = EDGE_V.get(char, "zh-CN-YunxiNeural")

                    async def _t():
                        c = edge_tts.Communicate(text, vn)
                        await c.save(out)

                    asyncio.run(_t())
                    ok = True
                except:
                    pass

            if not ok:
                _save_silence(out, float(dur))

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
