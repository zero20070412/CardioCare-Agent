"""
诊断推理引擎 - 集成 L-LSTrans 深度学习模型
支持 ECG/PCG 信号推理，输出特征向量和分类结果供 LLM 辅助诊断

两种运行模式:
1. 有 checkpoint → 真实推理 (需要 GPU 或 CPU)
2. 无 checkpoint → 模拟推理 (生成格式正确的演示数据)
"""

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ============================================================
# SNOMED-CT 标签映射 (31 种心血管疾病)
# ============================================================
LABEL_NAMES = {
    # CODE15 预训练 (6 类)
    "1dAVb": "一度房室传导阻滞",
    "RBBB": "右束支传导阻滞",
    "LBBB": "左束支传导阻滞",
    "SB": "窦性心动过缓",
    "ST": "ST 段改变",
    "AF": "心房颤动",
    # WFDB_Ga (18 类) - 与上面有交集
    "164889003": "心房颤动",
    "164890007": "心房扑动",
    "6374002": "束支传导阻滞",
    "426627000": "心动过缓",
    "733534002": "完全性左束支传导阻滞",
    "713427006": "完全性右束支传导阻滞",
    "270492004": "一度房室传导阻滞",
    "713426002": "不完全性右束支传导阻滞",
    "59118001": "右束支传导阻滞",
    "164909002": "左束支传导阻滞",
    "426783006": "正常窦性心律",
    "284470004": "房性早搏",
    "10370003": "起搏心律",
    "427172004": "室性早搏",
    # 通用
    "normal": "正常",
    "abnormal": "异常",
}


class DiagnosticInferenceEngine:
    """
    统一的深度学习诊断推理引擎

    使用方法:
        engine = DiagnosticInferenceEngine()
        engine.load_ecg_model(checkpoint_path="...", model_config="student")
        result = engine.analyze_ecg(ecg_signal)
    """

    def __init__(self):
        self._ecg_model = None
        self._pcg_model = None
        self._ecg_model_config = None
        self._pcg_model_config = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ecg_label_names = []
        self._pcg_label_names = ["正常", "异常"]

    @property
    def device(self):
        return self._device

    @property
    def has_ecg_model(self):
        return self._ecg_model is not None

    @property
    def has_pcg_model(self):
        return self._pcg_model is not None

    # ---- 模型加载 ----

    def load_ecg_model(self, checkpoint_path=None, model_config="student",
                       num_classes=6, num_leads=12, input_length=4096):
        """
        加载 ECG 诊断模型

        Args:
            checkpoint_path: 权重文件路径 (.pt)，None 时使用模拟推理
            model_config: "large"(42层/768维) / "light"(30层/128维) / "student"(9层/64维)
            num_classes: 分类类别数
            num_leads: 导联数 (12/3/1)
            input_length: 输入信号长度
        """
        try:
            from algorithms.deep_models.lstrans.model_zoo import LSTransECG

            config_map = {
                "large": (42, 768, 0.0001),
                "light": (30, 128, 0.001),
                "student": (9, 64, 0.001),
            }
            num_layers, complexity, lr = config_map.get(model_config, config_map["student"])

            self._ecg_model = LSTransECG(
                nOUT=num_classes,
                out_channels=complexity,
                in_channels=num_leads,
                input_length=input_length,
                num_layers=num_layers,
                rank_list=0,  # 无 LoRA，直接推理
            ).to(self._device)

            if checkpoint_path and os.path.isfile(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
                # 处理可能的 state_dict 包装
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                self._ecg_model.load_state_dict(state_dict, strict=False)
                logger.info(f"ECG 模型已从 {checkpoint_path} 加载")
            else:
                logger.info(f"未提供 ECG checkpoint，使用 {model_config} 配置的随机权重")

            self._ecg_model.eval()
            self._ecg_model_config = model_config
            self._ecg_label_names = self._get_ecg_label_names(num_classes)
            return True

        except Exception as e:
            logger.warning(f"ECG 模型加载失败: {e}，将使用模拟推理模式")
            self._ecg_model = None
            return False

    def load_pcg_model(self, checkpoint_path=None, model_config="student",
                       num_classes=2, input_length=40000, loc_dim=5):
        """
        加载 PCG 心音诊断模型

        Args:
            checkpoint_path: 权重文件路径 (.pt)，None 时使用模拟推理
            model_config: "large" / "light" / "student"
            num_classes: 分类类别数 (2: 正常/异常)
            input_length: 输入信号长度
            loc_dim: 听诊位置维度
        """
        try:
            from algorithms.deep_models.lstrans.model_zoo import LSTransPCG

            config_map = {
                "large": (42, 768),
                "light": (30, 128),
                "student": (9, 64),
            }
            num_layers, complexity = config_map.get(model_config, config_map["student"])

            self._pcg_model = LSTransPCG(
                nOUT=num_classes,
                out_channels=complexity,
                in_channels=1,
                input_length=input_length,
                num_layers=num_layers,
                rank_list=0,
                loc_dim=loc_dim,
            ).to(self._device)

            if checkpoint_path and os.path.isfile(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                self._pcg_model.load_state_dict(state_dict, strict=False)
                logger.info(f"PCG 模型已从 {checkpoint_path} 加载")
            else:
                logger.info(f"未提供 PCG checkpoint，使用 {model_config} 配置的随机权重")

            self._pcg_model.eval()
            self._pcg_model_config = model_config
            return True

        except Exception as e:
            logger.warning(f"PCG 模型加载失败: {e}，将使用模拟推理模式")
            self._pcg_model = None
            return False

    # ---- ECG 推理 ----

    def analyze_ecg(self, ecg_signal: np.ndarray = None) -> dict:
        """
        ECG 信号诊断推理

        Args:
            ecg_signal: numpy array [channels, length] 或 None (模拟数据)

        Returns:
            dict: {
                "module": "ecg_deep",
                "status": "success" | "demo",
                "model_config": "...",
                "features_summary": {...},
                "predictions": {...},
                "top_classes": [...],
                "risk_assessment": {...},
                "note": "..."
            }
        """
        if ecg_signal is not None:
            ecg_signal = np.array(ecg_signal, dtype=np.float32)

        if self._ecg_model is not None and ecg_signal is not None:
            return self._real_ecg_inference(ecg_signal)
        else:
            return self._demo_ecg_inference(ecg_signal)

    def _real_ecg_inference(self, ecg_signal: np.ndarray) -> dict:
        """真实 ECG 推理"""
        from algorithms.deep_models.preprocess_utils import preprocess_ecg_full

        try:
            # 预处理
            processed = preprocess_ecg_full(ecg_signal, target_length=4096)
            tensor = torch.from_numpy(processed).unsqueeze(0).float().to(self._device)

            with torch.no_grad():
                features = self._ecg_model.backbone.feature_extraction(tensor)
                logits = self._ecg_model(tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            # 提取特征统计
            feat_np = features.cpu().numpy()[0]
            feat_stats = {
                "dim": int(feat_np.shape[0]),
                "mean": float(np.mean(feat_np)),
                "std": float(np.std(feat_np)),
                "max": float(np.max(feat_np)),
                "min": float(np.min(feat_np)),
                "norm": float(np.linalg.norm(feat_np)),
            }

            # Top 分类
            sorted_idx = np.argsort(probs)[::-1]
            top_classes = []
            predictions = {}
            for idx in sorted_idx:
                label = self._ecg_label_names[idx] if idx < len(self._ecg_label_names) else f"class_{idx}"
                prob = float(probs[idx])
                predictions[label] = prob
                if len(top_classes) < 5 and prob > 0.05:
                    top_classes.append({"label": label, "probability": prob})

            # 风险评估
            risk_level = "low"
            high_risk_count = sum(1 for p in probs if p > 0.5)
            if high_risk_count >= 2:
                risk_level = "high"
            elif high_risk_count >= 1:
                risk_level = "moderate"

            note = self._generate_ecg_note(top_classes, risk_level, feat_stats)

            return {
                "module": "ecg_deep",
                "status": "success",
                "model_config": self._ecg_model_config,
                "features_summary": feat_stats,
                "predictions": predictions,
                "top_classes": top_classes,
                "risk_assessment": {
                    "risk_level": risk_level,
                    "detected_abnormalities": high_risk_count,
                    "signal_quality": "good" if feat_stats["norm"] > 0.1 else "poor",
                },
                "note": note,
            }

        except Exception as e:
            logger.error(f"ECG 推理失败: {e}")
            return self._demo_ecg_inference(ecg_signal, error=str(e))

    def _demo_ecg_inference(self, ecg_signal: np.ndarray = None, error: str = None) -> dict:
        """模拟 ECG 推理（用于测试和无 checkpoint 场景）"""
        if error:
            status = "demo"
            note = f"ECG 深度学习模型未加载或推理失败 ({error})，以下为演示结果。"
        else:
            status = "demo"
            note = "ECG 深度学习模型使用演示模式（未提供预训练权重），以下为模拟分析结果。"

        # 模拟特征统计
        feat_stats = {
            "dim": 64,
            "mean": round(float(np.random.normal(0.3, 0.15)), 6),
            "std": round(float(np.random.normal(0.5, 0.1)), 6),
            "max": round(float(np.random.normal(1.2, 0.3)), 6),
            "min": round(float(np.random.normal(-0.8, 0.3)), 6),
            "norm": round(float(np.random.normal(3.5, 1.0)), 6),
        }

        # 模拟分类结果
        demo_labels = ["心房颤动", "正常窦性心律", "窦性心动过缓",
                       "右束支传导阻滞", "ST 段改变", "一度房室传导阻滞"]
        demo_probs = np.random.dirichlet(np.ones(len(demo_labels)) * 0.3)

        predictions = {}
        top_classes = []
        for label, prob in zip(demo_labels, demo_probs):
            predictions[label] = round(float(prob), 4)

        sorted_pairs = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_pairs[:5]:
            top_classes.append({"label": label, "probability": prob})

        risk_level = "moderate" if demo_probs[0] > 0.4 else "low"

        return {
            "module": "ecg_deep",
            "status": status,
            "model_config": self._ecg_model_config or "student (demo)",
            "features_summary": feat_stats,
            "predictions": predictions,
            "top_classes": top_classes,
            "risk_assessment": {
                "risk_level": risk_level,
                "detected_abnormalities": sum(1 for p in demo_probs if p > 0.3),
                "signal_quality": "good",
            },
            "note": note,
        }

    # ---- PCG 推理 ----

    def analyze_pcg(self, pcg_signal: np.ndarray = None,
                    location: str = "MV") -> dict:
        """
        PCG 心音信号诊断推理

        Args:
            pcg_signal: numpy array [length] 或 None
            location: 听诊位置 (AV/PV/TV/MV/PhC)

        Returns:
            dict: 分析结果
        """
        if pcg_signal is not None:
            pcg_signal = np.array(pcg_signal, dtype=np.float32)

        if self._pcg_model is not None and pcg_signal is not None:
            return self._real_pcg_inference(pcg_signal, location)
        else:
            return self._demo_pcg_inference(pcg_signal, location)

    def _real_pcg_inference(self, pcg_signal: np.ndarray, location: str) -> dict:
        """真实 PCG 推理"""
        from algorithms.deep_models.preprocess_utils import preprocess_pcg

        loc_map = {"AV": 0, "PV": 1, "TV": 2, "MV": 3, "PhC": 4}
        try:
            # 预处理
            processed = preprocess_pcg(pcg_signal, target_length=40000)
            waveform_tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)

            # 位置编码 one-hot
            loc_onehot = np.zeros(5, dtype=np.float32)
            loc_idx = loc_map.get(location, 3)
            loc_onehot[loc_idx] = 1.0
            loc_tensor = torch.from_numpy(loc_onehot).unsqueeze(0)

            waveform_tensor = waveform_tensor.to(self._device)
            loc_tensor = loc_tensor.to(self._device)

            with torch.no_grad():
                # LSTransPCG.feature_extraction(x, loc) 接受信号+位置两个参数
                features = self._pcg_model.feature_extraction(waveform_tensor, loc_tensor)
                logits = self._pcg_model(waveform_tensor, loc_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            feat_np = features.cpu().numpy()[0]
            feat_stats = {
                "dim": int(feat_np.shape[0]),
                "mean": float(np.mean(feat_np)),
                "std": float(np.std(feat_np)),
                "norm": float(np.linalg.norm(feat_np)),
            }

            is_abnormal = probs[1] > probs[0]
            abnormality_score = float(probs[1])

            note = self._generate_pcg_note(is_abnormal, abnormality_score, location, feat_stats)

            return {
                "module": "pcg_deep",
                "status": "success",
                "model_config": self._pcg_model_config,
                "features_summary": feat_stats,
                "predictions": {
                    "正常": round(float(probs[0]), 4),
                    "异常": round(float(probs[1]), 4),
                },
                "top_classes": [
                    {"label": "正常", "probability": round(float(probs[0]), 4)},
                    {"label": "异常", "probability": round(float(probs[1]), 4)},
                ],
                "risk_assessment": {
                    "risk_level": "high" if abnormality_score > 0.6 else "moderate" if abnormality_score > 0.3 else "low",
                    "is_abnormal": is_abnormal,
                    "abnormality_score": abnormality_score,
                    "auscultation_location": location,
                },
                "note": note,
            }

        except Exception as e:
            logger.error(f"PCG 推理失败: {e}")
            return self._demo_pcg_inference(pcg_signal, location, error=str(e))

    def _demo_pcg_inference(self, pcg_signal: np.ndarray = None,
                            location: str = "MV", error: str = None) -> dict:
        """模拟 PCG 推理"""
        if error:
            status = "demo"
            note = f"PCG 深度学习模型未加载或推理失败 ({error})，以下为演示结果。"
        else:
            status = "demo"
            note = "PCG 深度学习模型使用演示模式（未提供预训练权重），以下为模拟分析结果。"

        feat_stats = {
            "dim": 80,
            "mean": round(float(np.random.normal(0.2, 0.1)), 6),
            "std": round(float(np.random.normal(0.4, 0.1)), 6),
            "norm": round(float(np.random.normal(2.8, 0.8)), 6),
        }

        abnormality_score = round(float(np.random.beta(2, 5)), 4)

        return {
            "module": "pcg_deep",
            "status": status,
            "model_config": self._pcg_model_config or "student (demo)",
            "features_summary": feat_stats,
            "predictions": {
                "正常": round(1 - abnormality_score, 4),
                "异常": abnormality_score,
            },
            "top_classes": [
                {"label": "正常", "probability": round(1 - abnormality_score, 4)},
                {"label": "异常", "probability": abnormality_score},
            ],
            "risk_assessment": {
                "risk_level": "high" if abnormality_score > 0.6 else "moderate" if abnormality_score > 0.3 else "low",
                "is_abnormal": abnormality_score > 0.5,
                "abnormality_score": abnormality_score,
                "auscultation_location": location,
            },
            "note": note,
        }

    # ---- 辅助方法 ----

    def _get_ecg_label_names(self, num_classes: int) -> list[str]:
        """根据类别数返回对应的中文标签名"""
        if num_classes == 6:
            return ["一度房室传导阻滞", "右束支传导阻滞", "左束支传导阻滞",
                    "窦性心动过缓", "ST 段改变", "心房颤动"]
        elif num_classes == 18:
            return ["心房颤动", "心房扑动", "束支传导阻滞", "心动过缓",
                    "完全性左束支传导阻滞", "完全性右束支传导阻滞",
                    "一度房室传导阻滞", "不完全性右束支传导阻滞",
                    "右束支传导阻滞", "左束支传导阻滞", "正常窦性心律",
                    "房性早搏", "起搏心律", "室性早搏",
                    "低 QRS 电压", "QRS 时限延长", "ST 段抬高", "T 波异常"]
        elif num_classes == 19:
            return ["心房颤动", "心房扑动", "束支传导阻滞", "心动过缓",
                    "完全性左束支传导阻滞", "完全性右束支传导阻滞",
                    "一度房室传导阻滞", "不完全性右束支传导阻滞",
                    "右束支传导阻滞", "左束支传导阻滞", "正常窦性心律",
                    "房性早搏", "起搏心律", "室性早搏",
                    "左前分支传导阻滞", "左轴偏移", "PR 间期延长",
                    "QT 间期延长", "QRS 低电压"]
        elif num_classes == 16:
            return ["心房颤动", "束支传导阻滞", "心动过缓", "一度房室传导阻滞",
                    "左束支传导阻滞", "不完全性右束支传导阻滞", "正常窦性心律",
                    "房性早搏", "室性早搏", "QRS 低电压", "ST 段改变",
                    "T 波异常", "T 波倒置", "PR 间期延长", "QT 间期延长",
                    "Q 波异常"]
        else:
            return [f"class_{i}" for i in range(num_classes)]

    @staticmethod
    def _generate_ecg_note(top_classes, risk_level, feat_stats):
        """生成 ECG 诊断说明文本"""
        top = top_classes[0] if top_classes else {"label": "未知", "probability": 0}
        lines = [
            f"基于 L-LSTrans 深度学习模型的心电分析:",
            f"最可能的诊断: {top['label']} (置信度 {top['probability']:.1%})",
        ]
        if len(top_classes) > 1:
            second = top_classes[1]
            lines.append(f"次要可能: {second['label']} (置信度 {second['probability']:.1%})")

        risk_desc = {"low": "低风险", "moderate": "中等风险", "high": "高风险"}
        lines.append(f"综合风险等级: {risk_desc.get(risk_level, '未知')}")

        lines.append(
            f"模型特征统计: 维度 {feat_stats['dim']}，"
            f"L2范数 {feat_stats['norm']:.3f}，"
            f"均值 {feat_stats['mean']:.3f}，标准差 {feat_stats['std']:.3f}"
        )

        if risk_level == "high":
            lines.append("建议: 检测到多项异常指标，建议尽快进行专业心电图检查并咨询心血管专科医生。")
        elif risk_level == "moderate":
            lines.append("建议: 存在部分异常信号，建议关注身体变化并定期复查。")
        else:
            lines.append("建议: 当前心电图信号未见明显异常，继续保持良好的心血管健康管理习惯。")

        return "\n".join(lines)

    @staticmethod
    def _generate_pcg_note(is_abnormal, score, location, feat_stats):
        """生成 PCG 诊断说明文本"""
        lines = ["基于 L-LSTrans 深度学习模型的心音分析:"]
        if is_abnormal:
            lines.append(f"检测结果: 异常 (异常概率 {score:.1%})")
            lines.append(f"听诊位置: {location}")
            lines.append("建议: 检测到心音异常信号，可能存在心脏杂音或瓣膜问题，建议进一步进行超声心动图检查。")
        else:
            lines.append(f"检测结果: 正常 (正常概率 {1 - score:.1%})")
            lines.append(f"听诊位置: {location}")
            lines.append("建议: 心音信号未见明显异常，S1/S2 心音正常。")

        lines.append(
            f"模型特征统计: 维度 {feat_stats['dim']}，"
            f"L2范数 {feat_stats['norm']:.3f}"
        )
        return "\n".join(lines)


# 全局推理引擎单例
_engine_instance: DiagnosticInferenceEngine | None = None


def get_inference_engine() -> DiagnosticInferenceEngine:
    """
    获取全局推理引擎单例。
    首次创建时自动从配置加载 ECG 和 PCG 预训练权重，
    加载失败时 graceful fallback 到 demo 模式。
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DiagnosticInferenceEngine()

        # 自动加载权重（仅在首次创建时）
        from utils.config import settings

        ecg_path = settings.ecg_checkpoint
        pcg_path = settings.pcg_checkpoint

        if ecg_path and os.path.isfile(ecg_path):
            _engine_instance.load_ecg_model(checkpoint_path=ecg_path, model_config="student")
        else:
            logger.debug(f"ECG 权重文件不存在 ({ecg_path})，使用 demo 模式")

        if pcg_path and os.path.isfile(pcg_path):
            _engine_instance.load_pcg_model(checkpoint_path=pcg_path, model_config="student")
        else:
            logger.debug(f"PCG 权重文件不存在 ({pcg_path})，使用 demo 模式")

    return _engine_instance
