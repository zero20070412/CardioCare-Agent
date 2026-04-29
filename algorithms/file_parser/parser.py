"""
文件解析模块 - 体检单 OCR + 信号数据读取
根据文件扩展名自动分发到对应的解析逻辑。
支持: PDF/JPG/PNG 体检报告, CSV/NPY/WAV 信号数据
"""

import base64
import logging
import os

import numpy as np
import pandas as pd
from typing import Any

logger = logging.getLogger(__name__)

# 文件类型分类
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PDF_TYPE = {".pdf"}
SIGNAL_TYPES = {".csv", ".npy", ".wav"}


def parse_uploaded_file(file_path: str) -> dict[str, Any]:
    """
    根据文件扩展名自动分发解析逻辑。

    Args:
        file_path: 上传文件的路径

    Returns:
        dict: {
            "status": "success" | "error",
            "file_type": "image" | "pdf" | "signal_csv" | "signal_npy" | "signal_wav",
            "category": "report" | "signal",
            "signal_type": "ecg" | "pcg" | "hrv" | None,
            "content": str | dict | list,  # 报告 base64 / 信号数据
            "filename": str,
        }
    """
    if not file_path or not os.path.isfile(file_path):
        return {"status": "error", "message": "无效文件路径"}

    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    if ext in IMAGE_TYPES:
        return _parse_image(file_path, filename)
    elif ext in PDF_TYPE:
        return _parse_pdf(file_path, filename)
    elif ext == ".csv":
        return _parse_csv(file_path, filename)
    elif ext == ".npy":
        return _parse_npy(file_path, filename)
    elif ext == ".wav":
        return _parse_wav(file_path, filename)
    else:
        return {"status": "error", "message": f"不支持的文件类型: {ext}"}


def _parse_image(file_path: str, filename: str) -> dict:
    """图片文件 -> base64，准备调用 Qwen-VL"""
    try:
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return {
            "status": "success",
            "file_type": "image",
            "category": "report",
            "signal_type": None,
            "content": f"data:image/jpeg;base64,{b64}",
            "filename": filename,
        }
    except Exception as e:
        return {"status": "error", "message": f"图片读取失败: {e}"}


def _parse_pdf(file_path: str, filename: str) -> dict:
    """PDF -> 使用 PyMuPDF 转图片 -> base64"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return {"status": "error", "message": "请安装 PyMuPDF: pip install PyMuPDF"}

    try:
        doc = fitz.open(file_path)
        images_base64 = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            images_base64.append(f"data:image/png;base64,{b64}")
        doc.close()

        return {
            "status": "success",
            "file_type": "pdf",
            "category": "report",
            "signal_type": None,
            "content": images_base64,
            "page_count": len(images_base64),
            "filename": filename,
        }
    except Exception as e:
        return {"status": "error", "message": f"PDF 解析失败: {e}"}


def _detect_signal_type(filename: str) -> str | None:
    """根据文件名猜测信号类型"""
    fn_lower = filename.lower()
    if "ecg" in fn_lower or "心电" in fn_lower:
        return "ecg"
    elif "pcg" in fn_lower or "心音" in fn_lower:
        return "pcg"
    elif "hrv" in fn_lower or "rr" in fn_lower:
        return "hrv"
    return None


def _parse_csv(file_path: str, filename: str) -> dict:
    """CSV 信号数据文件"""
    try:
        df = pd.read_csv(file_path, header=None)
        signal_type = _detect_signal_type(filename)

        # 提取数值数据
        data = df.values.flatten().tolist() if df.size > 0 else []

        return {
            "status": "success",
            "file_type": "signal_csv",
            "category": "signal",
            "signal_type": signal_type,
            "content": {"data": data, "columns": list(df.columns), "shape": list(df.shape)},
            "filename": filename,
        }
    except Exception as e:
        return {"status": "error", "message": f"CSV 解析失败: {e}"}


def _parse_npy(file_path: str, filename: str) -> dict:
    """numpy 信号数据文件"""
    try:
        arr = np.load(file_path)
        signal_type = _detect_signal_type(filename)
        return {
            "status": "success",
            "file_type": "signal_npy",
            "category": "signal",
            "signal_type": signal_type,
            "content": {"data": arr.tolist(), "shape": list(arr.shape)},
            "filename": filename,
        }
    except Exception as e:
        return {"status": "error", "message": f"NPY 解析失败: {e}"}


def _parse_wav(file_path: str, filename: str) -> dict:
    """WAV 信号数据文件（心音 PCG 或 HRV 来源）"""
    try:
        import soundfile as sf

        data, sr = sf.read(file_path, dtype="float32")
        if len(data.shape) == 2:
            data = data.mean(axis=1)

        signal_type = _detect_signal_type(filename)
        if signal_type is None:
            signal_type = "pcg"  # WAV 默认视为心音

        return {
            "status": "success",
            "file_type": "signal_wav",
            "category": "signal",
            "signal_type": signal_type,
            "content": {"data": data.tolist(), "sample_rate": sr, "length": len(data)},
            "filename": filename,
        }
    except Exception as e:
        return {"status": "error", "message": f"WAV 解析失败: {e}"}


__all__ = ["parse_uploaded_file", "_detect_signal_type"]
