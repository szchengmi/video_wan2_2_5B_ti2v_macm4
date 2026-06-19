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

    prompt = f"""你是一个专业的中文短剧编剧。请为一部{genre}题材的AI短剧写第{episode_num}集的完整剧本。

角色: 小明(28岁程序员,内向善良,戴眼镜短发) | 小丽(26岁设计师,活泼开朗,长发) | 王总(45岁总监,严厉公正)
场景: office(现代办公室) cafe(温馨咖啡馆) park(城市公园) apartment(温馨公寓) street(城市街道)

要求: 目标时长{duration_sec}秒, 每个镜头3-6秒, 合理安排场景数和镜头数, 完整故事线+悬念结尾

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
        return json.loads(text)
    except json.JSONDecodeError:
        # 修复截断JSON：补全未闭合的花括号
        fixed = text.rstrip().rstrip(',')
        opens = fixed.count('{')
        closes = fixed.count('}')
        if opens > closes:
            fixed += '}' * (opens - closes)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                return json.loads(m.group())
            # 最终降级：返回预置剧本
            return _get_fallback_script(episode_num, genre)


def _get_fallback_script(episode_num=1, genre="urban_romance", style="二次元"):
    """预置兜底剧本（Gemini JSON 解析失败时使用）"""
    return {
        "episode": episode_num,
        "title": "第一集：初遇",
        "style": style,
        "scenes": [
            {
                "scene_id": "scene_1", "location": "office", "time_of_day": "morning",
                "lighting": "自然光", "mood": "平静日常",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "小明在办公室敲代码", "character": "xiaoming",
                     "action": "专注地敲击键盘", "dialogue": "", "narration": "周一的早晨，办公室里只有键盘的声音。",
                     "emotion": "calm", "subtitle": "周一的早晨，办公室里只有键盘的声音。"},
                    {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 2, "description": "小明表情特写", "character": "xiaoming",
                     "action": "微微皱眉", "dialogue": "这个需求又改了...", "narration": "",
                     "emotion": "thoughtful", "subtitle": "这个需求又改了..."},
                    {"shot_id": "shot_3", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "小丽走进办公室", "character": "xiaoli",
                     "action": "推门进来，笑着打招呼", "dialogue": "早啊小明！今天天气真好！", "narration": "",
                     "emotion": "happy", "subtitle": "早啊小明！今天天气真好！"}
                ]
            },
            {
                "scene_id": "scene_2", "location": "cafe", "time_of_day": "afternoon",
                "lighting": "暖黄灯光", "mood": "温馨浪漫",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "小明和小丽在咖啡馆聊天", "character": "xiaoming",
                     "action": "端着咖啡杯，认真倾听", "dialogue": "你觉得这个设计方案怎么样？", "narration": "",
                     "emotion": "calm", "subtitle": "你觉得这个设计方案怎么样？"},
                    {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 2, "description": "小丽眼睛发亮", "character": "xiaoli",
                     "action": "眼睛发亮，兴奋地比划", "dialogue": "我觉得配色可以再大胆一些！", "narration": "",
                     "emotion": "happy", "subtitle": "我觉得配色可以再大胆一些！"},
                    {"shot_id": "shot_3", "shot_type": "medium_shot", "camera_movement": "pan_right",
                     "duration_seconds": 3, "description": "两人相视而笑", "character": "xiaoming",
                     "action": "忍不住笑了", "dialogue": "你总是这么有想法。", "narration": "",
                     "emotion": "embarrassed", "subtitle": "你总是这么有想法。"}
                ]
            },
            {
                "scene_id": "scene_3", "location": "office", "time_of_day": "evening",
                "lighting": "夕阳余晖", "mood": "紧张",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "王总走进办公室", "character": "boss_wang",
                     "action": "严肃地推门进来", "dialogue": "小明，客户对方案很不满意！", "narration": "",
                     "emotion": "angry", "subtitle": "小明，客户对方案很不满意！"},
                    {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 2, "description": "小明紧张的表情", "character": "xiaoming",
                     "action": "紧张地站起来", "dialogue": "什么？我明明按需求做的...", "narration": "",
                     "emotion": "nervous", "subtitle": "什么？我明明按需求做的..."},
                    {"shot_id": "shot_3", "shot_type": "wide_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "三人对峙", "character": "boss_wang",
                     "action": "将文件摔在桌上", "dialogue": "需求已经变了，你不知道吗？明天早上之前改好！", "narration": "",
                     "emotion": "angry", "subtitle": "需求已经变了，你不知道吗？明天早上之前改好！"}
                ]
            },
            {
                "scene_id": "scene_4", "location": "apartment", "time_of_day": "night",
                "lighting": "台灯光", "mood": "温馨感人",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "小明在公寓加班", "character": "xiaoming",
                     "action": "疲惫地盯着屏幕", "dialogue": "", "narration": "夜深了，小明还在改方案。",
                     "emotion": "sad", "subtitle": "夜深了，小明还在改方案。"},
                    {"shot_id": "shot_2", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "小丽端着夜宵进来", "character": "xiaoli",
                     "action": "轻轻推门，端着夜宵", "dialogue": "还没休息？我给你带了宵夜。", "narration": "",
                     "emotion": "calm", "subtitle": "还没休息？我给你带了宵夜。"},
                    {"shot_id": "shot_3", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 2, "description": "小明感动地看着小丽", "character": "xiaoming",
                     "action": "感动地看着小丽", "dialogue": "谢谢你，小丽。有你在真好。", "narration": "",
                     "emotion": "happy", "subtitle": "谢谢你，小丽。有你在真好。"}
                ]
            },
            {
                "scene_id": "scene_5", "location": "park", "time_of_day": "morning",
                "lighting": "阳光明媚", "mood": "充满希望",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "wide_shot", "camera_movement": "pan_left",
                     "duration_seconds": 3, "description": "公园里晨跑", "character": "xiaoming",
                     "action": "在公园晨跑", "dialogue": "", "narration": "改完方案的第二天，小明决定出门透透气。",
                     "emotion": "calm", "subtitle": "改完方案的第二天，小明决定出门透透气。"},
                    {"shot_id": "shot_2", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "偶遇小丽", "character": "xiaoli",
                     "action": "惊喜地挥手", "dialogue": "小明！好巧啊！", "narration": "",
                     "emotion": "surprised", "subtitle": "小明！好巧啊！"},
                    {"shot_id": "shot_3", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "两人并肩走在公园", "character": "xiaoming",
                     "action": "并肩散步，相视而笑", "dialogue": "小丽，昨晚的方案客户通过了！", "narration": "",
                     "emotion": "happy", "subtitle": "小丽，昨晚的方案客户通过了！"}
                ]
            },
            {
                "scene_id": "scene_6", "location": "office", "time_of_day": "morning",
                "lighting": "自然光", "mood": "紧张期待",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 3, "description": "王总宣布消息", "character": "boss_wang",
                     "action": "站在会议室前方", "dialogue": "告诉大家一个好消息——", "narration": "",
                     "emotion": "calm", "subtitle": "告诉大家一个好消息——"},
                    {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 2, "description": "小明和小丽紧张对视", "character": "xiaoming",
                     "action": "紧张地握紧拳头", "dialogue": "", "narration": "",
                     "emotion": "nervous", "subtitle": ""},
                    {"shot_id": "shot_3", "shot_type": "wide_shot", "camera_movement": "dolly_in",
                     "duration_seconds": 3, "description": "王总微笑", "character": "boss_wang",
                     "action": "露出罕见的微笑", "dialogue": "客户非常满意！小明、小丽，你们做到了！", "narration": "",
                     "emotion": "happy", "subtitle": "客户非常满意！小明、小丽，你们做到了！"}
                ]
            }
        ],
        "next_episode_hook": "小明和小丽的项目获得了成功，但新的挑战正在等着他们..."
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
