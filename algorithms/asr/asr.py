"""
ASR 语音转文字模块 - 基于 FunASR paraformer-zh
将语音文件转写为中文文字，支持 WAV/MP3 等常见格式。
"""

import logging
import os

logger = logging.getLogger(__name__)

# 全局 ASR 模型单例（延迟加载）
_asr_model = None


def _load_asr_model():
    """延迟加载 paraformer-zh 模型"""
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    try:
        from funasr import AutoModel

        logger.info("正在加载 paraformer-zh ASR 模型（首次加载可能需要几分钟）...")
        _asr_model = AutoModel(
            model="paraformer-zh",
            hub="ms",
            disable_update=True,
        )
        logger.info("ASR 模型加载完成。")
        return _asr_model
    except Exception as e:
        logger.error(f"ASR 模型加载失败: {e}")
        raise


def transcribe_audio(audio_path: str) -> dict:
    """
    将语音文件转写为文字。

    Args:
        audio_path: 语音文件路径（WAV/MP3 等）

    Returns:
        dict: {
            "status": "success" | "error",
            "text": str,           # 转写文字
            "message": str,
        }
    """
    if not audio_path or not os.path.isfile(audio_path):
        return {"status": "error", "text": "", "message": "无效的语音文件路径"}

    try:
        model = _load_asr_model()
        result = model.generate(input=audio_path)

        if not result or len(result) == 0:
            return {"status": "error", "text": "", "message": "ASR 模型未返回结果"}

        text = result[0].get("text", "").strip()
        if not text:
            return {"status": "error", "text": "", "message": "未能识别语音内容"}

        return {"status": "success", "text": text, "message": "转写成功"}

    except Exception as e:
        logger.error(f"ASR 转写失败: {e}")
        return {"status": "error", "text": "", "message": f"语音转写失败: {e}"}


__all__ = ["transcribe_audio"]
