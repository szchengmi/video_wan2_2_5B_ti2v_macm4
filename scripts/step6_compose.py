#!/usr/bin/env python3
"""Step 6: 剪辑合成 - FFmpeg"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

import argparse


def main():
    parser = argparse.ArgumentParser(description="AI短剧剪辑合成")
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--videos-dir", default="output/videos")
    parser.add_argument("--audio-dir", default="output/audio")
    parser.add_argument("--output-dir", default="output/final")
    args = parser.parse_args()

    sb = load_json(args.storyboard)
    dirs = get_dirs(sb.get("episode", 1))
    ep = sb.get("episode", 1)

    # 1. SRT
    srt = f"{args.output_dir}/ep{ep:02d}.srt"
    total_dur = _make_srt(sb, srt)

    # 2. 视频列表
    vids = []
    for scene in sb.get("scenes", []):
        for shot in scene.get("shots", []):
            sid = shot["shot_id"]
            vp = f"{args.videos_dir}/ep{ep:02d}_{scene['scene_id']}_{sid}.mp4"
            gp = vp.replace(".mp4", ".gif")
            if os.path.exists(vp):
                vids.append(vp)
            elif os.path.exists(gp):
                run_cmd(f'ffmpeg -y -i "{gp}" -c:v libx264 -pix_fmt yuv420p '
                        f'-movflags +faststart "{vp}" 2>/dev/null')
                if os.path.exists(vp):
                    vids.append(vp)

    # 3. 音频列表
    auds = []
    for scene in sb.get("scenes", []):
        for shot in scene.get("shots", []):
            sid = shot["shot_id"]
            ap = f"{args.audio_dir}/ep{ep:02d}_{scene['scene_id']}_{sid}.wav"
            if os.path.exists(ap):
                auds.append(ap)

    log(f"视频:{len(vids)} 音频:{len(auds)}")
    if not vids:
        log("[ERROR] 没有视频")
        return

    # 4. 拼接
    cv = f"{args.output_dir}/_v.mp4"
    ca = f"{args.output_dir}/_a.wav"
    _concat(vids, cv, "video")
    if auds:
        _concat(auds, ca, "audio")
    else:
        _save_silence(ca, total_dur)

    # 5. 合成
    final = f"{args.output_dir}/episode_{ep:02d}_final.mp4"
    cmd = (
        f'ffmpeg -y -i "{cv}" -i "{ca}" '
        f'-vf "subtitles=\'{srt}\':force_style=\'FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2\'" '
        f'-c:v libx264 -crf 20 -pix_fmt yuv420p '
        f'-c:a aac -b:a 128k -ar 44100 -ac 2 '
        f'-shortest -movflags +faststart "{final}"'
    )
    run_cmd(cmd, timeout=300)

    if os.path.exists(final):
        log(f"最终: {final} ({os.path.getsize(final)/1e6:.1f}MB)")
    else:
        # fallback
        cmd2 = (f'ffmpeg -y -i "{cv}" -i "{ca}" -c:v libx264 -crf 20 -pix_fmt yuv420p '
                f'-c:a aac -b:a 128k -ar 44100 -ac 2 -shortest -movflags +faststart "{final}"')
        run_cmd(cmd2, timeout=300)
        if os.path.exists(final):
            log(f"最终(无硬字幕): {final} ({os.path.getsize(final)/1e6:.1f}MB)")

    for f in [cv, ca]:
        if os.path.exists(f):
            os.remove(f)
    log("完成")


def _make_srt(sb, path):
    lines, idx, t = [], 1, 0.0
    for scene in sb.get("scenes", []):
        for shot in scene.get("shots", []):
            dur = shot.get("duration_seconds", 3)
            text = shot.get("subtitle") or shot.get("dialogue") or ""
            if text:
                lines.extend([str(idx), f"{seconds_to_srt_time(t)} --> {seconds_to_srt_time(t+dur)}", text, ""])
                idx += 1
            t += dur
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return t


def _concat(files, out, mtype):
    lf = out + ".list"
    with open(lf, "w") as f:
        for p in files:
            f.write(f"file '{p}'\n")
    if mtype == "video":
        cmd = f'ffmpeg -y -f concat -safe 0 -i "{lf}" -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "{out}"'
    else:
        cmd = f'ffmpeg -y -f concat -safe 0 -i "{lf}" -acodec pcm_s16le -ar {AUDIO_SAMPLE_RATE} -ac 1 "{out}"'
    run_cmd(cmd, timeout=300)
    if os.path.exists(lf):
        os.remove(lf)


if __name__ == "__main__":
    main()
