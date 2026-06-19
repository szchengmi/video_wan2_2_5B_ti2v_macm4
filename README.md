# video_wan2_2_5B_ti2v_macm4

Wan2.2 TI2V 5B AI 短剧 Pipeline — Mac mini M4 优化版

基于 [video_wan2_2_5B_ti2v](https://github.com/szchengmi/video_wan2_2_5B_ti2v) 适配的 Apple Silicon 本地运行版本。

## 特性

- **MPS 加速**: Apple Silicon 上自动启用 Metal GPU 加速
- **自动模型下载**: 运行 pipeline 时自动检测并下载缺失的模型文件
- **平台兼容**: Mac (MPS) / Kaggle (T4) / 其他 (CPU) 自动检测
- **6种视觉风格**: 二次元 / 古代田园 / 赛博朋克 / 动漫 / 类真人 / 火柴人
- **灵活时长**: `--duration 15` 指定秒数，镜头时长 3-6 秒由 LLM 动态分配

## Mac 环境要求

- macOS 14+ / Apple Silicon (M1/M2/M3/M4)
- 16GB+ 统一内存（推荐，Wan2.2 推理约需 8-10GB）
- 50GB+ 可用磁盘空间（模型 ~23.5GB）
- Python 3.10+

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/szchengmi/video_wan2_2_5B_ti2v_macm4.git
cd video_wan2_2_5B_ti2v_macm4/scripts

# 2. 安装依赖
pip install torch torchaudio  # MPS 支持的 PyTorch
pip install -r requirements.txt

# 3. 设置 API Key
export GOOGLE_API_KEY="你的Gemini Key"
export HF_TOKEN="你的 HuggingFace Token"  # 可选，加速模型下载

# 4. 运行 pipeline
python kaggle_pipeline.py --force --duration 15 --style 二次元
```

## CLI 参数

```bash
python kaggle_pipeline.py [OPTIONS]
  --force           强制重新生成
  --episode N       集数 (默认: 1)
  --duration SECS   目标时长/秒 (默认: 15)
  --style STYLE     视频风格

可选风格:
  二次元    (默认) 日漫风格，活力色彩，吉卜力感
  古代田园  国画水墨，稻田山水，淡彩
  赛博朋克  霓虹灯，暗城，Blade Runner
  动漫      日漫标准，精致阴影，表现力
  类真人    真人质感，电影灯光，浅景深
  火柴人    极简线条，白底黑线，无阴影
```

## 模型下载

模型文件会自动下载到 `~/heipiworkspace/mac/projects/video_wan2_2_5B_ti2v_macm4/models/`。

也可手动运行：

```bash
python download_models.py
```

模型清单:
| 模型 | 文件 | 大小 |
|------|------|------|
| UNET 5B | `wan2.2_ti2v_5B_fp16.safetensors` | ~10GB |
| CLIP (umt5) | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | ~12GB |
| VAE | `wan2.2_vae.safetensors` | ~1.5GB |

## Pipeline 架构

```
Step 1: 剧本生成 (Gemini API)
Step 2: 分镜生成 (结构化 JSON + 风格注入)
Step 3: 跳过 (Wan2.2 直接 T2V)
Step 4: 视频生成 (Wan2.2 TI2V 5B via ComfyUI + MPS)
Step 5: 配音生成 (ChatTTS / edge-tts)
Step 6: 剪辑合成 (FFmpeg)
```

## 目录结构

```
video_wan2_2_5B_ti2v_macm4/
├── scripts/
│   ├── kaggle_pipeline.py      # 主入口
│   ├── common.py               # 公共配置 + 风格预设
│   ├── download_models.py       # 模型下载脚本
│   ├── step1_generate_story.py # 剧本生成
│   ├── step2_generate_storyboard.py # 分镜生成
│   ├── step4_generate_videos_wan22.py # 视频生成
│   ├── step5_generate_audio.py # 配音
│   └── step6_compose.py       # 合成
├── models/                     # 模型文件 (自动下载)
├── output/                     # 输出 (自动生成)
└── workflow.json               # ComfyUI 工作流定义
```

## License

MIT
