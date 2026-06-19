#!/usr/bin/env python3
"""Step 2: 分镜生成 - 从剧本生成分镜JSON"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

import argparse


def generate_storyboard(script_data, style_name=None):
    # 风格: 优先参数 > storyboard顶层字段 > 默认
    if style_name is None:
        style_name = script_data.get("style", DEFAULT_STYLE)
    style = STYLE_PRESETS.get(style_name, STYLE_PRESETS[DEFAULT_STYLE])

    storyboard = {
        "episode": script_data.get("episode", 1),
        "title": script_data.get("title", ""),
        "style": style_name,
        "characters": {},
        "scenes": []
    }
    characters_used = set()

    for scene in script_data.get("scenes", []):
        scene_data = {
            "scene_id": scene["scene_id"],
            "location": scene["location"],
            "time_of_day": scene["time_of_day"],
            "lighting": scene["lighting"],
            "mood": scene["mood"],
            "shots": []
        }
        for shot in scene.get("shots", []):
            char = shot.get("character", "none")
            # 兼容 character 是列表的情况 (如 ["xiaoming"])
            if isinstance(char, list):
                char = char[0] if char else "none"
            shot_type = shot.get("shot_type", "medium_shot")
            emotion = shot.get("emotion", "calm")
            if char != "none":
                characters_used.add(char)
            params = SHOT_PARAMS.get(shot_type, SHOT_PARAMS["medium_shot"])

            pp = [params["prefix"]]
            if char != "none" and char in CHARACTER_PROMPTS:
                pp.append(CHARACTER_PROMPTS[char]["base_prompt"])
            pp.append(shot.get("action", ""))
            if emotion in EMOTION_ENHANCE:
                pp.append(EMOTION_ENHANCE[emotion])
            sp = SCENE_PROMPTS.get(scene["location"], "")
            if sp:
                pp.append(sp)
            # 注入风格化的场景后缀
            pp.append(style["scene_suffix"])
            pp.append(f"{scene['time_of_day']}, {scene['lighting']}")
            pp.append(shot.get("description", ""))
            pp.append("masterpiece, best quality, detailed")
            pp.append(style["prompt_suffix"])
            full_prompt = ", ".join([p for p in pp if p])

            neg = style["negative_prompt"]
            if char != "none" and char in CHARACTER_PROMPTS:
                neg += ", " + CHARACTER_PROMPTS[char]["negative_prompt"]

            scene_data["shots"].append({
                "shot_id": shot["shot_id"],
                "shot_type": shot_type,
                "camera_movement": shot.get("camera_movement", "static"),
                "duration_seconds": shot.get("duration_seconds", 3),
                "width": params["w"],
                "height": params["h"],
                "prompt": full_prompt,
                "negative_prompt": neg,
                "seed": CHARACTER_PROMPTS.get(char, {}).get("seed", -1),
                "character": char,
                "dialogue": shot.get("dialogue", ""),
                "narration": shot.get("narration", ""),
                "subtitle": shot.get("subtitle", ""),
                "description": shot.get("description", ""),
                "action": shot.get("action", ""),
                "emotion": emotion,
                "steps": IMAGE_STEPS,
                "guidance": IMAGE_GUIDANCE
            })
        storyboard["scenes"].append(scene_data)

    # 角色名映射: 中文 → 拼音 (CHARACTER_PROMPTS 的 key)
    CHAR_NAME_MAP = {
        "小明": "xiaoming", "小丽": "xiaoli", "王总": "boss_wang",
        "xiaoming": "xiaoming", "xiaoli": "xiaoli", "boss_wang": "boss_wang",
    }

    for cid in characters_used:
        mapped = CHAR_NAME_MAP.get(cid, cid)
        if mapped in CHARACTER_PROMPTS:
            storyboard["characters"][cid] = CHARACTER_PROMPTS[mapped]
        else:
            # fallback: 用通用配置
            storyboard["characters"][cid] = {
                "base_prompt": f"{cid}, anime style, high quality",
                "negative_prompt": "ugly, deformed, blurry, low quality",
                "seed": 42,
            }
    return storyboard


def main():
    parser = argparse.ArgumentParser(description="AI短剧分镜生成")
    parser.add_argument("--script", required=True, help="剧本JSON路径")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    script_data = load_json(args.script)
    style_name = script_data.get("style", None)
    storyboard = generate_storyboard(script_data, style_name=style_name)
    save_json(storyboard, f"{args.output_dir}/episode_{storyboard['episode']:02d}_storyboard.json")
    total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))
    print(f"[OK] {len(storyboard['scenes'])}场景 | {total}镜头")


if __name__ == "__main__":
    main()
