#!/usr/bin/env python3
"""Step 1: 剧本生成 - Gemini API"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

import argparse
import json
import re


def generate_script(episode_num=1, genre="urban_romance", prev_summary="",
                    duration_minutes=3, style="二次元"):
    from google import genai

    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Gemini 根据总时长灵活决定场景数和镜头数
    # 每个镜头 duration_seconds 在剧本中指定，总时长 ≈ 所有镜头之和
    duration_sec = int(duration_minutes)  # 参数名保留但实际是秒

    prompt = f"""你是一个专业的中文短剧编剧和声音设计师。请为一部{genre}题材的AI短剧写第{episode_num}集的完整剧本。

角色: 小明(28岁程序员,内向善良,戴眼镜短发) | 小丽(26岁设计师,活泼开朗,长发) | 王总(45岁总监,严厉公正)
场景: office(现代办公室) cafe(温馨咖啡馆) park(城市公园) apartment(温馨公寓) street(城市街道)

═══════════════════════════════════════════
节奏规划原则（重要！）
═══════════════════════════════════════════
1. 镜头时长由内容决定，不是固定值：
   - 对话密集场景：4-8 秒（让观众看清画面+听清对话）
   - 纯环境/过渡镜头：2-4 秒
   - 动作/紧张场景：3-5 秒，可快速切换
   - 建立镜头（每场第一个）：稍长 4-6 秒
2. 场景规划：
   - 不要频繁换场景，一个场景内可以有多个长镜头
   - 用不同景别讲述故事：建立→对话→反应→特写，避免无意义切镜
   - 总镜头数由 Gemini 根据故事节奏自然决定，不要刻意凑数也不要刻意限制
   - 长镜头优先：一个镜头可以持续 6-12 秒，给观众沉浸感
   - 只有当视角/地点/时间确实需要变化时才切换镜头
3. 留白与呼吸：
   - 不要填满整个 duration，留 10-15% 给沉默/留白
   - 对话之间有 0.3-0.8 秒的停顿间隔
   - 镜头切换处留 0.2-0.5 秒黑场过渡

═══════════════════════════════════════════
音频设计原则（重要！）
═══════════════════════════════════════════
每个镜头必须设计声音层次，用 audio_events 对象详细描述：

audio_events 包含 4 个子数组（必须全部存在，无内容写空数组 []）：

1. dialogue（角色对话）：有角色台词时填写
   - role: 角色拼音 (xiaoming/xiaoli/boss_wang)
   - lines: 台词内容
   - timecode: "开始秒-结束秒" 格式，如 "0.5-2.3"
   - emotion: 情绪（参考下方情绪词）
   - speed: 语速偏移，正数=快，负数=慢，如 +3/-1/0

2. VO（旁白）：有旁白时填写
   - role: 固定 "narrator"
   - lines: 旁白内容
   - timecode: 时间范围
   - emotion: 情绪
   - speed: 语速偏移

3. SFX（音效）：有音效时填写（如雷声、关门、脚步声、手机响等）
   - sound: 音效名称标识
   - path: 音效文件路径，如 "sound/door_close.mp3"
   - timecode: 时间范围
   - volume: 音量 0.0-1.0

4. Atmos（环境音）：有环境音时填写（如街道噪音、雨声、室内空调声等）
   - sound: 环境音名称标识
   - path: 音效文件路径，如 "sound/rain.mp3"
   - timecode: 时间范围（通常覆盖整个镜头）
   - volume: 音量 0.0-1.0
   - loop: 是否循环播放 true/false

时间码规则：
- 从 0.0 开始计算，第一个声音元素从 0.0 或稍后开始
- 对话之间留 0.3-0.8 秒间隔
- 所有时间码不能超过 duration_seconds
- 最后一个声音结束后留 0.3-0.5 秒静音

情绪词库：happy/sad/angry/surprised/nervous/calm/determined/embarrassed/thoughtful/紧张/感激/抒情/平静/恐惧/愤怒/温柔/急迫

duration_seconds 计算：
- Gemini 根据音频时间码自行计算：duration = max(所有事件结束时间) + 0.5s 留白
- 确保 duration 合理（2-8秒之间），与镜头内容匹配
- 总时长约 {duration_sec} 秒（所有镜头之和）

═══════════════════════════════════════════
纯JSON输出（不要 markdown 代码块）
═══════════════════════════════════════════
shots 结构: {{"shot_id","shot_type","camera_movement","duration_seconds","description","character","action","emotion","audio_events","subtitle"}}
audio_events 结构: {{"dialogue":[{{"role","lines","timecode","emotion","speed"}}],"VO":[...],"SFX":[...],"Atmos":[...]}}
示例: {{"episode":1,"title":"标题","style":"{style}","scenes":[{{"scene_id":"scene_1","location":"office","time_of_day":"morning","lighting":"自然光","mood":"氛围","shots":[{{"shot_id":"shot_1","shot_type":"medium_shot","camera_movement":"static","duration_seconds":5.0,"description":"画面描述","character":"xiaoming","action":"动作","emotion":"calm","audio_events":{{"dialogue":[{{"role":"xiaoming","lines":"台词","timecode":"0.5-2.3","emotion":"紧张","speed":0}}],"VO":[],"SFX":[],"Atmos":[]}},"subtitle":"字幕"}}],"next_hook":"下集预告"}}]}}
"""


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
        config={
            "temperature": 0.9, "top_p": 0.95, "top_k": 40, "max_output_tokens": 65536,
        },
    )
    text = response.text.strip()
    if not text:
        print(f"[GEMINI] 返回空文本，fallback")
        return _get_fallback_script(episode_num, genre, duration_sec, style)
    print(f"[GEMINI] 返回前200字: {text[:200]}")
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    try:
        script = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[GEMINI] JSON解析失败: {e}")
        print(f"[GEMINI] 后200字: ...{text[-200:]}")
    else:
        return _fix_script(script, duration_sec)
    # 修复截断JSON：补全未闭合的花括号和方括号
    fixed = text.rstrip().rstrip(',')
    opens_b = fixed.count('{')
    closes_b = fixed.count('}')
    opens_k = fixed.count('[')
    closes_k = fixed.count(']')
    if opens_b > closes_b:
        fixed += '}' * (opens_b - closes_b)
    if opens_k > closes_k:
        fixed += ']' * (opens_k - closes_k)
    try:
        script = json.loads(fixed)
    except json.JSONDecodeError:
        pass
    else:
        return _fix_script(script, duration_sec)
    # 尝试用正则提取最长的完整JSON对象
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            script = json.loads(m.group())
        except json.JSONDecodeError:
            pass
        else:
            return _fix_script(script, duration_sec)
    # 最终降级：返回预置剧本
    return _get_fallback_script(episode_num, genre, duration_sec, style)


def _fix_script(script, target_duration):
    """修正剧本：确保每个镜头有 audio_events 结构，并修正总时长"""
    scenes = script.get("scenes", [])
    if not scenes:
        return _get_fallback_script(script.get("episode", 1), "urban_romance", target_duration, script.get("style", "二次元"))

    # 确保每个镜头有 audio_events 结构
    for sc in scenes:
        for shot in sc.get("shots", []):
            if "audio_events" not in shot or not isinstance(shot.get("audio_events"), dict):
                # 从旧字段构造 audio_events
                audio_events = {"dialogue": [], "VO": [], "SFX": [], "Atmos": []}
                old_dialogue = shot.get("dialogue", "")
                old_narration = shot.get("narration", "")
                char = shot.get("character", "xiaoming")
                emotion = shot.get("emotion", "calm")
                dur = shot.get("duration_seconds", 5)
                if old_dialogue and isinstance(old_dialogue, str) and old_dialogue.strip():
                    audio_events["dialogue"].append({
                        "role": char if char != "none" else "xiaoming",
                        "lines": old_dialogue.strip(),
                        "timecode": f"0.5-{min(dur-0.5, 3.0):.1f}",
                        "emotion": emotion,
                        "speed": 0
                    })
                if old_narration and isinstance(old_narration, str) and old_narration.strip():
                    audio_events["VO"].append({
                        "role": "narrator",
                        "lines": old_narration.strip(),
                        "timecode": f"0.3-{min(dur-0.5, 4.0):.1f}",
                        "emotion": "平静",
                        "speed": -1
                    })
                shot["audio_events"] = audio_events

    # 修正总时长（最后一个镜头吸收误差）
    actual_total = sum(s.get("duration_seconds", 5) for sc in scenes for s in sc.get("shots", []))
    if actual_total != target_duration and actual_total > 0:
        diff = target_duration - actual_total
        last_shot = scenes[-1]["shots"][-1]
        new_dur = last_shot["duration_seconds"] + diff
        new_dur = max(2, min(8, new_dur))
        last_shot["duration_seconds"] = new_dur

    return script




def _get_fallback_script(episode_num=1, genre="urban_romance", duration_sec=8, style="二次元"):
    """预置兜底剧本（Gemini JSON 解析失败时使用），每个镜头包含 audio_events"""
    scenes = []
    remaining = duration_sec
    scene_idx = 1
    locations = ["office", "cafe", "park", "apartment", "street"]
    times = ["morning", "afternoon", "evening", "night", "morning"]

    while remaining > 0 and scene_idx <= 6:
        dur = min(remaining, 6)
        audio_events = {
            "dialogue": [{"role": "xiaoming", "lines": "怎么了？", "timecode": f"0.3-{min(dur-0.5, 2.5):.1f}", "emotion": "calm", "speed": 0}],
            "VO": [],
            "SFX": [],
            "Atmos": []
        }
        shot = {
            "shot_id": f"shot_1", "shot_type": "medium_shot", "camera_movement": "static",
            "duration_seconds": dur, "description": f"场景{scene_idx}镜头", "character": "xiaoming",
            "action": "动作", "emotion": "calm",
            "audio_events": audio_events, "subtitle": ""
        }
        scenes.append({
            "scene_id": f"scene_{scene_idx}",
            "location": locations[(scene_idx - 1) % 5],
            "time_of_day": times[(scene_idx - 1) % 5],
            "lighting": "自然光", "mood": "平静",
            "shots": [shot]
        })
        remaining -= dur
        scene_idx += 1

    return {
        "episode": episode_num,
        "title": f"第{episode_num}集：短剧",
        "style": style,
        "scenes": scenes,
        "next_episode_hook": "故事继续..."
    }


def main():
    parser = argparse.ArgumentParser(description="AI短剧剧本生成")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--genre", default="urban_romance")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    if not GOOGLE_API_KEY:
        print("[ERROR] 设置 GOOGLE_API_KEY 环境变量")
        return

    print(f"生成第{args.episode}集剧本...")
    script = generate_script(args.episode, args.genre)

    os.makedirs(args.output_dir, exist_ok=True)
    path = f"{args.output_dir}/episode_{args.episode:02d}_script.json"
    save_json(script, path)
    print(f"[OK] {path} | {script.get('title')}")
    return script


if __name__ == "__main__":
    main()
