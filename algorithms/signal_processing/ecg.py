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


def _generate_synthetic_ecg_signal(
    num_leads: int = 12,
    length: int = 4096,
    heart_rate: float = 72,
    noise_std: float = 0.02,
) -> np.ndarray:
    """
    生成仿真 ECG 信号用于模型推理（无真实数据时的 fallback）。

    使用叠加正弦波模拟 P-QRS-T 波形，各导联有微小差异。
    返回 shape (num_leads, length) 的 float32 numpy array。
    """
    fs = 250  # 采样率
    t = np.arange(length, dtype=np.float32) / fs
    beat_period = 60.0 / heart_rate  # 单个心动周期 (秒)

    signal = np.zeros((num_leads, length), dtype=np.float32)

    for lead_idx in range(num_leads):
        # 每个导联的幅度和相位略有不同（模拟空间差异）
        amp_scale = 1.0 + 0.15 * np.sin(lead_idx * 0.5)
        phase_offset = lead_idx * 0.02

        lead_signal = np.zeros(length, dtype=np.float32)

        for beat_start in np.arange(0, t[-1], beat_period):
            # 以每个 R 波峰为中心构建局部波形
            r_center = beat_start + phase_offset
            local_t = t - r_center  # 相对于 R 峰的时间

            # P 波 (心房去极化)
            p_mask = (local_t >= -0.28) & (local_t <= -0.12)
            lead_signal += np.where(
                p_mask,
                0.15 * amp_scale * np.sin(np.pi * (local_t + 0.28) / 0.16),
                0.0,
            )

            # Q 波
            q_mask = (local_t >= -0.08) & (local_t <= -0.04)
            lead_signal += np.where(
                q_mask,
                -0.1 * amp_scale * np.sin(np.pi * (local_t + 0.08) / 0.04),
                0.0,
            )

            # R 波 (QRS 复合波主峰)
            r_mask = (local_t >= -0.04) & (local_t <= 0.04)
            lead_signal += np.where(
                r_mask,
                1.0 * amp_scale * np.sin(np.pi * (local_t + 0.04) / 0.08),
                0.0,
            )

            # S 波
            s_mask = (local_t >= 0.04) & (local_t <= 0.08)
            lead_signal += np.where(
                s_mask,
                -0.2 * amp_scale * np.sin(np.pi * (local_t - 0.04) / 0.04),
                0.0,
            )

            # T 波 (心室复极化)
            t_mask = (local_t >= 0.16) & (local_t <= 0.36)
            lead_signal += np.where(
                t_mask,
                0.3 * amp_scale * np.sin(np.pi * (local_t - 0.16) / 0.20),
                0.0,
            )

        # 添加基线漂移和噪声
        baseline = 0.03 * np.sin(2 * np.pi * 0.15 * t + lead_idx)
        noise = np.random.normal(0, noise_std, length).astype(np.float32)

        signal[lead_idx] = lead_signal + baseline + noise

    return signal


def analyze_ecg(
    signal_data: Sequence[float] | None = None,
    sampling_rate: int = 250,
) -> dict[str, object]:
    """
    ECG 信号分析 - 集成深度学习模型推理

    如果提供了信号数据且模型已加载，会进行真实推理；
    如果模型已加载但无信号数据，自动生成仿真信号进行推理；
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
    data_is_synthetic = False
    try:
        from algorithms.deep_models.inference_engine import get_inference_engine

        engine = get_inference_engine()

        if engine.has_ecg_model:
            if signal_data is not None and len(signal_data) > 100:
                # 有真实信号数据 → 真实推理
                sig_array = np.array(signal_data, dtype=np.float32)
                if sig_array.ndim == 1:
                    total_len = len(sig_array)
                    n_leads = 12
                    lead_len = total_len // n_leads
                    sig_2d = sig_array[:lead_len * n_leads].reshape(n_leads, lead_len)
                else:
                    sig_2d = sig_array
                deep_result = engine.analyze_ecg(sig_2d)
            else:
                # 模型已加载但无真实数据 → 生成仿真信号进行推理
                synthetic_signal = _generate_synthetic_ecg_signal()
                deep_result = engine.analyze_ecg(synthetic_signal)
                data_is_synthetic = True
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
            "synthetic": data_is_synthetic,
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
