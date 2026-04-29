"""
ECG 信号分析模块 - 集成 L-LSTrans 深度学习模型
支持真实 ECG 信号推理，保留 mock 波形作为 fallback
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _generate_mock_ecg_waveform(num_points: int = 240) -> list[float]:
    """Generate a deterministic placeholder ECG-like waveform."""
    waveform: list[float] = []
    for index in range(num_points):
        phase = (index % 40) / 40
        baseline = 0.03 * math.sin(index / 6)

        if phase < 0.1:
            value = baseline + 0.05 * math.sin(phase * math.pi * 10)
        elif phase < 0.16:
            value = baseline - 0.08
        elif phase < 0.2:
            value = baseline + 1.0
        elif phase < 0.26:
            value = baseline - 0.12
        elif phase < 0.45:
            value = baseline + 0.18 * math.sin((phase - 0.26) * math.pi * 5)
        else:
            value = baseline

        waveform.append(round(value, 4))
    return waveform


def analyze_ecg(
    signal_data: Sequence[float] | None = None,
    sampling_rate: int = 250,
) -> dict[str, object]:
    """
    ECG 信号分析 - 集成深度学习模型推理

    如果提供了信号数据且模型已加载，会进行真实推理；
    否则使用 mock 结果。

    Args:
        signal_data: ECG 信号数据 (展平的一维列表或 2D numpy array)
        sampling_rate: 采样率 (Hz)

    Returns:
        dict: ECG 分析结果
    """
    waveform = list(signal_data) if signal_data else _generate_mock_ecg_waveform()

    # 尝试调用深度学习模型
    deep_result = None
    try:
        from algorithms.deep_models.inference_engine import get_inference_engine

        engine = get_inference_engine()
        if signal_data is not None and len(signal_data) > 100:
            # 将展平的信号转为 2D numpy array
            sig_array = np.array(signal_data, dtype=np.float32)
            if sig_array.ndim == 1:
                # 假设 12 导联，均匀分割
                total_len = len(sig_array)
                n_leads = 12
                lead_len = total_len // n_leads
                sig_2d = sig_array[:lead_len * n_leads].reshape(n_leads, lead_len)
            else:
                sig_2d = sig_array

            deep_result = engine.analyze_ecg(sig_2d)
    except Exception as e:
        logger.debug(f"深度学习 ECG 推理跳过: {e}")

    # 构建基础结果
    result = {
        "module": "ecg",
        "status": deep_result.get("status", "placeholder") if deep_result else "placeholder",
        "algorithm": "L-LSTrans ECG 深度学习模型" if deep_result and deep_result.get("status") == "success" else "ECG 分析 (Pan-Tompkins 算法预留)",
        "summary": deep_result.get("note", "ECG 分析完成。稳定的窦性心律 mock 结果已返回。") if deep_result else "ECG 分析完成。稳定的窦性心律 mock 结果已返回。",
        "risk_level": "low",
        "metrics": {
            "heart_rate_bpm": 72,
            "rhythm": "regular",
            "r_peak_count": 6,
            "signal_quality": "good",
        },
        "visualization": {
            "waveform": waveform,
            "sampling_rate": sampling_rate,
        },
        "meta": {
            "input_provided": bool(signal_data),
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
        result["meta"]["deep_model_loaded"] = deep_result.get("status") == "success"
        result["meta"]["placeholder_only"] = deep_result.get("status") != "success"

    return result


__all__ = ["analyze_ecg"]
