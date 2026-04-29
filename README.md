# CardioBot - AI 心血管健康管理助手

<p align="center">
  <strong>基于 Gradio和大语言模型的多模态心血管健康评估与主动干预系统</strong>
</p>

---

## 项目简介

CardioBot 是一款面向个人用户的心血管健康管理 AI 助手，集成多模态信号分析、语音情感识别和智能对话能力，帮助用户进行日常心血管健康监测与风险评估。

### 核心能力

| 模块                 | 功能                                                         | 状态 |
| -------------------- | ------------------------------------------------------------ | ---- |
| **智能对话**         | 基于大语言模型的心血管健康咨询，支持文字与语音交互           |
| **语音情感识别**     | emotion2vec_plus_seed 9 类情感识别，结合压力评估触发主动干预 |
| **ASR 语音转文字**   | FunASR paraformer-zh 中文语音识别                            |
| **体检单 OCR**       | Qwen-VL 视觉模型识别 PDF/图片体检报告                        |
| **ECG 心电分析**     | L-LSTrans 深度学习模型推理（需模型权重）                     |
| **PCG 心音分析**     | L-LSTrans 深度学习模型推理（需模型权重）                     |
| **HRV 心率变异分析** | RR 间期统计指标计算（SDNN/RMSSD/pNN50）                      |
| **呼吸训练干预**     | 基于压力等级生成个性化呼吸方案                               |
| **音乐疗法推荐**     | 基于情感和压力推荐个性化音乐                                 |
| **多模态融合评估**   | ECG + HRV + PCG 综合健康评分                                 |

## 技术栈

- **前端**: Gradio 6.x
- **LLM**: 通义千问
- **视觉模型**: Qwen-VL
- **ASR**: FunASR paraformer-zh
- **语音情感**: emotion2vec_plus_seed
- **深度模型**: L-LSTrans
- **语言**: Python 3.10+

## 项目结构

```
cardiobot-main/
├── app.py                        # 主入口，Gradio 界面启动
├── requirements.txt              # Python 依赖
├── .env.example                  # 环境变量模板
├── .gitignore
│
├── agent/                        # 智能体核心
│   ├── core.py                   # 主调度
│   ├── model.py                  # LLM API 封装
│   ├── prompts.py                # 系统提示词
│   ├── memory.py                 # 会话记忆管理
│   └── history_manager.py        # 历史记录持久化
│
├── algorithms/                   # 算法模块
│   ├── signal_processing/        # 生理信号分析
│   │   ├── ecg.py                # ECG 心电分析
│   │   ├── pcg.py                # PCG 心音分析
│   │   └── hrv.py                # HRV 心率变异分析
│   ├── emotion_recognition/
│   │   └── voice.py              # 语音情感识别
│   ├── deep_models/
│   │   ├── cnn.py                # CNN 应激评估
│   │   ├── transformer.py        # 多模态 Transformer 融合
│   │   ├── inference_engine.py   # 统一推理引擎
│   │   └── lstrans/              # L-LSTrans 模型代码
│   │       ├── lsnet_se.py       # ECG 分支网络
│   │       ├── ska_ecg.py        # 可分离核注意力
│   │       ├── lora_layer.py     # LoRA 微调层
│   │       └── model_zoo.py      # 模型注册
│   ├── asr/
│   │   └── asr.py                # 语音转文字
│   ├── file_parser/
│   │   └── parser.py             # 文件解析
│   └── intervention/
│       ├── breathing.py          # 呼吸训练方案
│       └── music.py              # 音乐疗法推荐
│
├── frontend/                     # Gradio UI 组件
│   ├── chat_panel.py             # 聊天区域
│   ├── chart_panel.py            # HRV / 应激压力面板
│   └── sidebar.py               # 侧边栏
│
├── utils/
│   └── config.py                 # 环境变量读取
│
└── tests/                        # 测试
    ├── test_emotion2vec_integration.py
    ├── test_lstrans_integration.py
    ├── test_new_features.py
    └── test_algorithms_mock.py
```

## 快速开始

### 1. 环境准备

# 搭建虚拟环境

# 安装依赖

pip install -r requirements.txt

````

### 2. 配置环境变量

```bash
cp .env.example .env
````

编辑 `.env` 文件，填入你的 API 配置：

```env
# 使用真实模型时设为 false
USE_MOCK_MODEL=false

# DashScope API（通义千问）
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen3.6-flash

# 视觉模型（体检单 OCR，可选）
VISION_MODEL_NAME=qwen-vl-max
```

### 3. 启动

```bash
python app.py
```

浏览器访问 `http://localhost:7860` 即可使用。

### Mock 模式

不配置 API Key 时系统自动进入 Mock 模式，所有 LLM 回复为模拟文本，适合前端开发和功能测试。

## 支持的文件格式

| 类型     | 格式          | 用途                    |
| -------- | ------------- | ----------------------- |
| 体检报告 | PDF, JPG, PNG | OCR 识别后交由 LLM 分析 |
| 心电信号 | CSV, NPY      | ECG 深度学习模型推理    |
| 心音信号 | WAV, CSV, NPY | PCG 深度学习模型推理    |
| 心率变异 | CSV, NPY      | HRV 统计分析            |

## 工作流程

```
用户输入（文字 / 语音 / 文件上传）
        │
        ├── 文字消息 ──→ LLM 对话回复
        │
        ├── 语音消息 ──→ ASR 转文字 → 情感识别 → LLM 对话回复
        │                           │
        │                     极端压力时触发：
        │                     ├── 呼吸训练干预
        │                     └── 音乐疗法推荐
        │
        └── 文件上传
              ├── 体检单（PDF/图片）→ OCR → LLM 分析
              └── 信号数据（ECG/PCG/HRV）→ 算法分析 → 结果注入 LLM → 综合回复
```

## 开发状态

- **已完成**: 智能对话系统、语音交互链路、文件解析、主动干预、前端界面
- **待完善**: ECG/PCG 深度模型需加载训练好的权重、HRV 统计计算待实现、多模态融合对齐层为占位，HRV指标计算模型
