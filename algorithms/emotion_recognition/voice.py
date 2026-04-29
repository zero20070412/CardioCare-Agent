"""
语音情感识别模块 - 基于 emotion2vec 模型
使用 FunASR + emotion2vec_plus_seed 实现真实的语音情感分析
"""

import logging
import os

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

# emotion2vec 9 类情感标签
EMOTION2VEC_LABELS = [
    "生气/angry",
    "厌恶/disgusted",
    "恐惧/fearful",
    "开心/happy",
    "中立/neutral",
    "其他/other",
    "难过/sad",
    "吃惊/surprised",
    "<unk>",
]

# 中文情感名称映射
EMOTION_CN = {
    "生气/angry": "愤怒",
    "厌恶/disgusted": "厌恶",
    "恐惧/fearful": "恐惧",
    "开心/happy": "开心",
    "中立/neutral": "平静",
    "其他/other": "其他",
    "难过/sad": "悲伤",
    "吃惊/surprised": "惊讶",
    "<unk>": "未知",
}

# 情感到压力等级的映射
EMOTION_STRESS_MAP = {
    "生气/angry": "high",
    "厌恶/disgusted": "high",
    "恐惧/fearful": "high",
    "开心/happy": "low",
    "中立/neutral": "low",
    "其他/other": "moderate",
    "难过/sad": "moderate",
    "吃惊/surprised": "moderate",
    "<unk>": "unknown",
}

# 全局模型单例（延迟加载）
_model = None


def _load_model():
    """延迟加载 emotion2vec 模型（全局单例）"""
    global _model
    if _model is not None:
        return _model

    try:
        from funasr import AutoModel

        logger.info("正在加载 emotion2vec_plus_seed 模型（首次加载可能需要几分钟）...")
        _model = AutoModel(
            model="iic/emotion2vec_plus_seed",
            hub="ms",
            disable_update=True,
        )
        logger.info("emotion2vec 模型加载完成。")
        return _model
    except Exception as e:
        logger.error(f"emotion2vec 模型加载失败: {e}")
        raise


def _load_audio_as_tensor(audio_path: str):
    """用 soundfile 加载音频文件并转为 torch tensor"""
    speech, sample_rate = sf.read(audio_path, dtype="float32")

    # 如果是立体声，转为单声道
    if len(speech.shape) == 2:
        speech = speech.mean(axis=1)

    # emotion2vec 需要 16kHz，如果采样率不匹配则重采样
    if sample_rate != 16000:
        import torchaudio.transforms as T

        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        speech_tensor = torch.from_numpy(speech).float()
        speech_tensor = resampler(speech_tensor)
    else:
        speech_tensor = torch.from_numpy(speech).float()

    return speech_tensor


def analyze_voice_emotion(audio_path=None):
    """
    分析语音文件的情感状态

    Args:
        audio_path: 语音文件路径（WAV 格式），支持任意采样率

    Returns:
        dict: 包含情感分析结果的字典
            - status: "success" 或 "error"
            - emotion: 检测到的主要情感（中文）
            - emotion_en: 英文情感标签
            - stress_level: 压力等级 (low/moderate/high)
            - confidence: 置信度
            - scores: 所有 9 类情感的得分字典
            - note: 人类可读的分析说明
    """
    if audio_path is None or not os.path.isfile(audio_path):
        return {
            "status": "error",
            "emotion": "未知",
            "emotion_en": "unknown",
            "stress_level": "unknown",
            "confidence": 0.0,
            "scores": {},
            "note": "未提供有效的语音文件路径。",
        }

    try:
        model = _load_model()
        speech_tensor = _load_audio_as_tensor(audio_path)

        # 调用 emotion2vec 推理
        result = model.generate(
            speech_tensor,
            granularity="utterance",
            extract_embedding=False,
            fs=16000,
        )

        if not result or len(result) == 0:
            return {
                "status": "error",
                "emotion": "未知",
                "emotion_en": "unknown",
                "stress_level": "unknown",
                "confidence": 0.0,
                "scores": {},
                "note": "模型未返回有效结果。",
            }

        item = result[0]
        labels = item.get("labels", EMOTION2VEC_LABELS)
        scores_list = item.get("scores", [])

        # 构建得分字典
        scores = {}
        top_label = labels[0]
        top_score = 0.0
        for i, (label, score) in enumerate(zip(labels, scores_list)):
            scores[label] = round(float(score), 6)
            if float(score) > top_score:
                top_score = float(score)
                top_label = label

        # 提取结果
        emotion_en = top_label
        emotion_cn = EMOTION_CN.get(top_label, top_label)
        stress_level = EMOTION_STRESS_MAP.get(top_label, "unknown")
        confidence = round(top_score, 4)

        # 生成分析说明
        note = _generate_note(emotion_cn, stress_level, confidence, scores)

        return {
            "status": "success",
            "emotion": emotion_cn,
            "emotion_en": emotion_en,
            "stress_level": stress_level,
            "confidence": confidence,
            "scores": scores,
            "note": note,
        }

    except Exception as e:
        logger.error(f"语音情感分析失败: {e}")
        return {
            "status": "error",
            "emotion": "未知",
            "emotion_en": "unknown",
            "stress_level": "unknown",
            "confidence": 0.0,
            "scores": {},
            "note": f"语音情感分析出错: {str(e)}",
        }


def _generate_note(emotion_cn, stress_level, confidence, scores):
    """根据分析结果生成人类可读的说明"""
    if stress_level == "high":
        stress_desc = "较高"
        suggestion = "建议尝试深呼吸放松或聆听舒缓音乐来缓解情绪。"
    elif stress_level == "moderate":
        stress_desc = "中等"
        suggestion = "可以适当进行呼吸训练，帮助身心恢复平静。"
    else:
        stress_desc = "较低"
        suggestion = "当前情绪状态良好，继续保持。"

    # 找出第二高分的情感
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    secondary = ""
    if len(sorted_emotions) >= 2 and sorted_emotions[1][1] > 0.05:
        secondary_cn = EMOTION_CN.get(sorted_emotions[1][0], sorted_emotions[1][0])
        secondary = f"，同时带有轻微的{secondary_cn}情绪"

    return (
        f"检测到您当前的主要情绪为【{emotion_cn}】，"
        f"置信度 {confidence:.1%}，"
        f"压力水平{stress_desc}{secondary}。{suggestion}"
    )
