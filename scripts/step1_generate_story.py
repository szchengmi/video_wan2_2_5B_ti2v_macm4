#!/usr/bin/env python3
"""Step 1: 剧本生成 - Gemini API"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

import argparse
import json
import re


def _calc_scene_limits(duration_sec):
    """根据时长返回场景数量限制说明（每个场景约3-5秒）"""
    if duration_sec <= 3:
        return "1个场景"
    elif duration_sec <= 6:
        return "1-2个场景"
    elif duration_sec <= 8:
        return "2个场景"  # 8秒最佳：2场景各4秒
    elif duration_sec <= 10:
        return "2-3个场景"
    elif duration_sec <= 15:
        return "3个场景"
    else:
        scenes = duration_sec // 4
        return f"{max(1, scenes-1)}-{scenes+1}个场景"


def generate_script(episode_num=1, genre="urban_romance", prev_summary="",
                    duration_minutes=3, style="二次元"):
    from google import genai

    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Gemini 根据总时长灵活决定场景数和镜头数
    # 每个镜头 duration_seconds 在剧本中指定，总时长 ≈ 所有镜头之和
    duration_sec = int(duration_minutes)  # 参数名保留但实际是秒

    prompt = f"""你是一个专业的中文短剧编剧。请为一部{genre}题材的AI短剧写第{episode_num}集的完整剧本。

角色: 小明(28岁程序员,内向善良,戴眼镜短发) | 小丽(26岁设计师,活泼开朗,长发) | 王总(45岁总监,严厉公正)
场景: office(现代办公室) cafe(温馨咖啡馆) park(城市公园) apartment(温馨公寓) street(城市街道)

要求（必须严格遵守，否则输出无效）:
- 总时长严格等于{duration_sec}秒，所有镜头 duration_seconds 之和必须正好等于{duration_sec}
- 每个镜头 duration_seconds 在 2-4 秒之间（最少2秒，最多4秒）
- 每个场景包含 1-2 个镜头，每个场景总时长约 3-5 秒
- 场景数量严格限制：{duration_sec}秒 → {_calc_scene_limits(duration_sec)}
- 场景数 = 镜头数（每个镜头一个场景）
- 完整故事线+悬念结尾
- 严禁超过或低于目标时长
- 示例：8秒 → 2个场景(各1个镜头4秒) 或 1个场景(2个镜头各4秒) 或 2个场景(第1场景2镜头共5秒+第2场景2镜头共3秒)
- 不允许多余镜头或场景：场景数和镜头数必须正好符合上述规则
- **文本字数控制**：
  - 用于生成音频的最终文本字数 = duration_seconds × 4（中文语速约4字/秒）
  - 优先使用 dialogue（角色对话）；dialogue 为空则用 narration（旁白）
  - 总字数严格控制在 duration_seconds × 4 以内，保证配音时长与视频时长匹配
  - dialogue 和 narration 不能同时为空

纯JSON输出:
{{"episode": {episode_num}, "title": "标题", "style": "{style}", "scenes": [{{"scene_id": "scene_1", "location": "office",
"time_of_day": "morning", "lighting": "自然光", "mood": "氛围",
"shots": [{{"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
"duration_seconds": 3, "description": "画面描述", "character": "xiaoming",
"action": "动作", "dialogue": "对话", "narration": "旁白",
"emotion": "情绪", "subtitle": "字幕"}}]}}], "next_episode_hook": "下集预告"}}"""


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
        config={
            "temperature": 0.9, "top_p": 0.95, "top_k": 40, "max_output_tokens": 16384,
        },
    )
    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    try:
        script = json.loads(text)
    except json.JSONDecodeError:
        pass
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
    """修正剧本：确保总时长、场景数、镜头数符合约束"""
    scenes = script.get("scenes", [])
    if not scenes:
        return _get_fallback_script(script.get("episode", 1), "urban_romance", target_duration, script.get("style", "二次元"))

    # 计算每个场景的总时长和实际镜头数
    scene_durations = []
    for sc in scenes:
        shots = sc.get("shots", [])
        total = sum(s.get("duration_seconds", 3) for s in shots)
        scene_durations.append({"scene": sc, "shots": shots, "duration": total})

    actual_total = sum(sd["duration"] for sd in scene_durations)

    # 1. 如果总时长不对，按比例缩放每个镜头
    if actual_total != target_duration and actual_total > 0:
        ratio = target_duration / actual_total
        for sd in scene_durations:
            new_total = 0
            for i, shot in enumerate(sd["shots"]):
                if i == len(sd["shots"]) - 1:
                    # 最后一个镜头吸收误差
                    sd["shots"][i]["duration_seconds"] = max(2, min(4, target_duration - new_total))
                    new_total += sd["shots"][i]["duration_seconds"]
                else:
                    new_dur = max(2, min(4, round(shot.get("duration_seconds", 3) * ratio)))
                    sd["shots"][i]["duration_seconds"] = new_dur
                    new_total += new_dur
            sd["duration"] = new_total

    # 2. 修正场景数：8秒目标 → 2个场景
    expected_scenes = _target_scene_count(target_duration)
    actual_scenes = len(scene_durations)

    if actual_scenes > expected_scenes:
        # 场景太多：合并相邻场景到目标数量
        while len(scene_durations) > expected_scenes:
            # 找到相邻两个最短场景合并
            min_combined = float("inf")
            merge_idx = 0
            for i in range(len(scene_durations) - 1):
                combined = scene_durations[i]["duration"] + scene_durations[i + 1]["duration"]
                if combined < min_combined:
                    min_combined = combined
                    merge_idx = i
            # 合并
            merged_shots = scene_durations[merge_idx]["shots"] + scene_durations[merge_idx + 1]["shots"]
            merged_scene = scene_durations[merge_idx]["scene"]
            merged_scene["shots"] = merged_shots
            merged_scene["scene_id"] = f"scene_{merge_idx + 1}"
            scene_durations[merge_idx] = {"scene": merged_scene, "shots": merged_shots, "duration": min_combined}
            del scene_durations[merge_idx + 1]

    elif actual_scenes < expected_scenes and actual_scenes >= 1:
        # 场景太少：拆分最长场景
        while len(scene_durations) < expected_scenes:
            # 找最长场景
            longest_idx = max(range(len(scene_durations)), key=lambda i: scene_durations[i]["duration"])
            sd = scene_durations[longest_idx]
            if len(sd["shots"]) < 2:
                break  # 只有一个镜头无法拆
            # 拆分成两组
            mid = len(sd["shots"]) // 2
            shots_a = sd["shots"][:mid]
            shots_b = sd["shots"][mid:]
            dur_a = sum(s.get("duration_seconds", 3) for s in shots_a)
            dur_b = sum(s.get("duration_seconds", 3) for s in shots_b)
            # 创建新场景
            new_scene = dict(sd["scene"])
            new_scene["shots"] = shots_b
            new_scene["scene_id"] = f"scene_{longest_idx + 2}"
            sd["scene"]["shots"] = shots_a
            sd["scene"]["scene_id"] = f"scene_{longest_idx + 1}"
            sd["shots"] = shots_a
            sd["duration"] = dur_a
            scene_durations.insert(longest_idx + 1, {"scene": new_scene, "shots": shots_b, "duration": dur_b})

    # 3. 重新编号 scene_id
    for i, sd in enumerate(scene_durations):
        sd["scene"]["scene_id"] = f"scene_{i + 1}"

    # 4. 重新计算并修正总时长误差
    final_total = sum(sd["duration"] for sd in scene_durations)
    diff = target_duration - final_total
    if diff != 0:
        # 把误差分配到最后一个镜头的 duration_seconds
        last_scene = scene_durations[-1]
        last_shot = last_scene["shots"][-1]
        new_dur = last_shot["duration_seconds"] + diff
        if new_dur < 2:
            new_dur = 2
        elif new_dur > 4:
            new_dur = 4
        last_shot["duration_seconds"] = new_dur
        last_scene["duration"] = sum(s.get("duration_seconds", 3) for s in last_scene["shots"])

    script["scenes"] = [sd["scene"] for sd in scene_durations]
    return script


def _target_scene_count(duration_sec):
    """返回目标场景数（每个场景约3-5秒）"""
    if duration_sec <= 3:
        return 1
    elif duration_sec <= 6:
        return 2
    elif duration_sec <= 8:
        return 2
    elif duration_sec <= 10:
        return 3
    elif duration_sec <= 15:
        return 3
    else:
        return max(2, duration_sec // 4)


def _get_fallback_script(episode_num=1, genre="urban_romance", duration_sec=8, style="二次元"):
    """预置兜底剧本（Gemini JSON 解析失败时使用），动态适配目标时长"""
    target_scenes = _target_scene_count(duration_sec)
    # 每个场景平均时长
    avg_per_scene = duration_sec / target_scenes
    scenes = []
    # 预置对话模板（兜底用），按镜头时长×4 控制字数
    _fallback_dialogues = [
        "今天天气真好啊", "你怎么来了", "没关系，我来帮你",
        "等一下，我想说", "我们走吧", "太不可思议了",
        "你在看什么", "明天见", "我相信你",
        "这不可能", "谢谢你", "别担心",
        "快点过来", "今天加班吗", "晚餐吃什么",
        "看那边", "快躲开", "他在哪",
        "我找到了", "别放手", "快跑",
        "终于到了", "别说了", "等等我",
    ]
    _diag_idx = 0

    def _get_fallback_dur(duration_sec):
        """生成兜底对话，字数 = duration_sec × 4"""
        nonlocal _diag_idx
        target_len = duration_sec * 4
        # 拼接短语直到达到目标字数
        text = ""
        while len(text) < target_len:
            text += _fallback_dialogues[_diag_idx % len(_fallback_dialogues)]
            _diag_idx += 1
        # 精确截断到目标字数
        # 在最后一个标点处截断以保持通顺
        if len(text) > target_len:
            # 往回找最近的标点
            cut = target_len
            for offset in range(min(cut, 10)):
                if text[cut - offset] in "，。！？；":
                    cut = cut - offset
                    break
            text = text[:cut]
        return text

    for i in range(target_scenes):
        if i == target_scenes - 1:
            # 最后一个场景吸收误差
            scene_dur = duration_sec - sum(s.get("_dur", 0) for s in scenes)
        else:
            scene_dur = round(avg_per_scene)
        # 每个场景 1-2 个镜头
        if scene_dur <= 3:
            shots = [{"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                      "duration_seconds": scene_dur, "description": f"场景{i+1}镜头", "character": "xiaoming",
                      "action": "动作", "dialogue": _get_fallback_dur(scene_dur),
                      "narration": "", "emotion": "calm", "subtitle": ""}]
        else:
            d1 = scene_dur // 2
            d2 = scene_dur - d1
            shots = [
                {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                 "duration_seconds": max(2, min(4, d1)), "description": f"场景{i+1}镜头1", "character": "xiaoming",
                 "action": "动作", "dialogue": _get_fallback_dur(max(2, min(4, d1))),
                 "narration": "", "emotion": "calm", "subtitle": ""},
                {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                 "duration_seconds": max(2, min(4, d2)), "description": f"场景{i+1}镜头2", "character": "xiaoli",
                 "action": "动作", "dialogue": _get_fallback_dur(max(2, min(4, d2))),
                 "narration": "", "emotion": "calm", "subtitle": ""}
            ]
        scenes.append({
            "scene_id": f"scene_{i+1}",
            "location": ["office", "cafe", "park", "apartment", "street"][i % 5],
            "time_of_day": ["morning", "afternoon", "evening", "night", "morning"][i % 5],
            "lighting": "自然光", "mood": "平静",
            "shots": shots,
            "_dur": sum(s["duration_seconds"] for s in shots)
        })
    # 清理内部字段
    for s in scenes:
        del s["_dur"]
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
