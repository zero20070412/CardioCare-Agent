"""
信号预处理工具 - 从 MultiModeforHeart 项目适配
支持 ECG 带通滤波、StandardScaler 归一化、PCG Z-score 标准化
"""

import numpy as np
from scipy import signal as sig
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

# ECG 预处理配置
ECG_PREPROCESS_CFG = {
    "filter_bandpass": [1.0, 47.0],
    "filter_highpass": [],
    "filter_lowpass": [],
    "filter_notch": [],
    "scaler": "standard",
}

# PCG 预处理配置
PCG_TARGET_LENGTH = 40000
PCG_SAMPLE_RATE = 4000


def filter_signal(x: np.ndarray, sample_rate: int = 500) -> np.ndarray:
    """
    ECG 信号滤波: 带通 1-47Hz (PhysioNet 标准)

    Args:
        x: numpy array, shape [channels, length]
        sample_rate: 采样率 (Hz)

    Returns:
        滤波后的信号, shape 不变
    """
    cfg = ECG_PREPROCESS_CFG
    nyq = sample_rate * 0.5
    x = x.copy()

    for i in range(len(x)):
        # 高通滤波
        for cutoff in cfg["filter_highpass"]:
            if cutoff < nyq:
                b, a = sig.butter(3, cutoff / nyq, btype="highpass")
                x[i, :] = sig.filtfilt(b, a, x[i, :])
        # 低通滤波
        for cutoff in cfg["filter_lowpass"]:
            cutoff = min(cutoff, nyq - 0.05)
            b, a = sig.butter(3, cutoff / nyq, btype="lowpass")
            x[i, :] = sig.filtfilt(b, a, x[i, :])
        # 带通滤波 (主要使用 1-47Hz)
        if len(cfg["filter_bandpass"]) == 2:
            low, high = cfg["filter_bandpass"]
            if high >= nyq:
                high = nyq - 0.05
            b, a = sig.butter(3, [low / nyq, high / nyq], btype="bandpass")
            x[i, :] = sig.filtfilt(b, a, x[i, :])
        # 陷波滤波 (50Hz 工频干扰)
        for cutoff in cfg["filter_notch"]:
            if cutoff < nyq:
                b, a = sig.iirnotch(cutoff / nyq, 30)
                x[i, :] = sig.filtfilt(b, a, x[i, :])

    return x


def scale_signal(x: np.ndarray) -> np.ndarray:
    """
    ECG 信号标准化: 每个导联独立 StandardScaler

    Args:
        x: numpy array, shape [channels, length]

    Returns:
        标准化后的信号
    """
    x = x.copy()
    for i in range(len(x)):
        scaler = StandardScaler()
        x[i, :] = scaler.fit_transform(x[i, :].reshape(-1, 1)).squeeze()
    return x


def pad_or_crop_ecg(x: np.ndarray, target_length: int = 4096) -> np.ndarray:
    """
    ECG 信号裁剪/填充至目标长度

    Args:
        x: numpy array, shape [channels, length] 或 [channels, 1, length]
        target_length: 目标长度

    Returns:
        shape [channels, target_length]
    """
    if x.ndim == 3:
        x = np.squeeze(x)
    channels, length = x.shape
    if length < target_length:
        padding = np.zeros((channels, target_length - length), dtype=np.float32)
        x = np.concatenate((x, padding), axis=1)
    elif length > target_length:
        x = x[:, :target_length]
    return x.astype(np.float32)


def preprocess_pcg(waveform: np.ndarray, target_length: int = PCG_TARGET_LENGTH,
                   is_train: bool = False) -> np.ndarray:
    """
    PCG 心音信号预处理: Z-score 标准化 + 裁剪/填充

    Args:
        waveform: 1D numpy array
        target_length: 目标长度 (默认 40000 = 10s @ 4kHz)
        is_train: 是否为训练模式 (训练时随机裁剪，测试时居中裁剪)

    Returns:
        预处理后的 1D numpy array, shape [target_length]
    """
    # Z-score 标准化
    mu = np.mean(waveform)
    sigma = np.std(waveform) + 1e-8
    waveform = (waveform - mu) / sigma
    # 裁剪异常值
    waveform = np.clip(waveform, -3, 3)
    # 裁剪/填充
    if len(waveform) > target_length:
        if is_train:
            start = np.random.randint(0, len(waveform) - target_length)
        else:
            start = (len(waveform) - target_length) // 2
        waveform = waveform[start: start + target_length]
    else:
        pad_width = target_length - len(waveform)
        waveform = np.pad(waveform, (0, pad_width), mode="constant")
    return waveform.astype(np.float32)


def preprocess_ecg_full(ecg_signal: np.ndarray, sample_rate: int = 500,
                        target_length: int = 4096) -> np.ndarray:
    """
    完整的 ECG 预处理管线: 裁剪/填充 → 滤波 → 标准化

    Args:
        ecg_signal: numpy array, shape [channels, length]
        sample_rate: 采样率
        target_length: 目标长度

    Returns:
        预处理后的信号, shape [channels, target_length]
    """
    ecg_signal = pad_or_crop_ecg(ecg_signal, target_length)
    ecg_signal = filter_signal(ecg_signal, sample_rate)
    ecg_signal = scale_signal(ecg_signal)
    return ecg_signal
