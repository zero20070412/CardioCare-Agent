"""
PCG 心音分析模块 - 集成 L-LSTrans 深度学习模型
支持真实心音信号推理，保留 mock 波形作为 fallback
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _generate_mock_pcg_waveform(num_points: int = 240) -> list[float]:
    """Generate a deterministic placeholder PCG-like waveform."""
    waveform: list[float] = []
    for index in range(num_points):
        phase = (index % 60) / 60
        baseline = 0.01 * math.sin(index / 8)

        if 0.08 <= phase < 0.14:
            value = baseline + 0.65 * math.sin((phase - 0.08) * math.pi * 16)
        elif 0.36 <= phase < 0.43:
            value = baseline + 0.45 * math.sin((phase - 0.36) * math.pi * 14)
        else:
            value = baseline

        waveform.append(round(value, 4))
    return waveform


def analyze_pcg(
    audio_data: Sequence[float] | None = None,
    sampling_rate: int = 2000,
) -> dict[str, object]:
    """
    PCG 心音分析 - 集成深度学习模型推理

    如果提供了信号数据且模型已加载，会进行真实推理；
    否则使用 mock 结果。

    Args:
        audio_data: PCG 心音信号数据 (一维列表)
        sampling_rate: 采样率 (Hz)

    Returns:
        dict: PCG 分析结果
    """
    waveform = list(audio_data) if audio_data else _generate_mock_pcg_waveform()

    # 尝试调用深度学习模型
    deep_result = None
    try:
        from algorithms.deep_models.inference_engine import get_inference_engine

        engine = get_inference_engine()
        if audio_data is not None and len(audio_data) > 100:
            sig_array = np.array(audio_data, dtype=np.float32)
            deep_result = engine.analyze_pcg(sig_array, location="MV")
    except Exception as e:
        logger.debug(f"深度学习 PCG 推理跳过: {e}")

    # 构建基础结果
    result = {
        "module": "pcg",
        "status": deep_result.get("status", "placeholder") if deep_result else "placeholder",
        "algorithm": "L-LSTrans PCG 深度学习模型" if deep_result and deep_result.get("status") == "success" else "心音分析 (预留)",
        "summary": deep_result.get("note", "PCG 分析完成。正常 S1/S2 mock 结果已返回。") if deep_result else "PCG 分析完成。正常 S1/S2 mock 结果已返回。",
        "risk_level": "low",
        "metrics": {
            "heart_sound_pattern": "normal_s1_s2",
            "murmur_risk": "low",
            "abnormal_sound_detected": False,
            "signal_quality": "good",
        },
        "visualization": {
            "waveform": waveform,
            "sampling_rate": sampling_rate,
        },
        "meta": {
            "input_provided": bool(audio_data),
            "placeholder_only": True,
        },
    }

    # 合并深度学习结果
    if deep_result:
        risk = deep_result.get("risk_assessment", {})
        result["risk_level"] = risk.get("risk_level", "low")
        result["deep_analysis"] = {
            "model_config": deep_result.get("model_config"),
            "features_summary": deep_result.get("features_summary"),
            "predictions": deep_result.get("predictions"),
            "top_classes": deep_result.get("top_classes"),
        }
        result["metrics"]["abnormal_sound_detected"] = risk.get("is_abnormal", False)
        result["metrics"]["murmur_risk"] = risk.get("risk_level", "low")
        result["meta"]["deep_model_loaded"] = deep_result.get("status") == "success"
        result["meta"]["placeholder_only"] = deep_result.get("status") != "success"

    return result


__all__ = ["analyze_pcg"]
