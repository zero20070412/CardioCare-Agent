from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping, Sequence

from agent.memory import MemoryStore, default_memory_store
from agent.model import LLMClient, ModelClientError, ModelResponse, default_model_client
from algorithms.deep_models.cnn import analyze_cnn
from algorithms.deep_models.transformer import analyze_fusion
from algorithms.signal_processing.ecg import analyze_ecg
from algorithms.signal_processing.hrv import analyze_hrv
from algorithms.signal_processing.pcg import analyze_pcg


DEFAULT_SYSTEM_PROMPT = (
    "你是一个心血管健康状态评估与主动干预智能助手。"
    "请结合用户描述、历史对话和算法分析结果，给出清晰、谨慎、易理解的回复。"
    "你可以解释 ECG、PCG、HRV 和多模态占位分析结果，但必须明确这些结果当前仅用于系统联调与辅助说明。"
    "当涉及健康风险时，请提醒用户该系统不能替代专业医生诊断。"
)

ALGORITHM_KEYWORDS = {
    "ecg": ("ecg", "心电", "心电图", "心率", "心律", "波形", "心电分析"),
    "pcg": ("pcg", "心音", "杂音", "心音分析"),
    "hrv": ("hrv", "心率变异", "压力", "恢复"),
    "cnn": ("cnn", "应激", "stress", "风险", "深度诊断"),
    "fusion": ("融合", "多模态", "综合", "transformer", "综合评估"),
}


def _run_voice_emotion_analysis(audio_path):
    """
    执行语音情感分析，并根据结果触发主动干预（呼吸训练 + 音乐推荐）。
    返回包含语音情感结果和干预建议的列表。
    """
    from algorithms.emotion_recognition.voice import analyze_voice_emotion
    from algorithms.intervention.breathing import generate_breathing_guidance
    from algorithms.intervention.music import recommend_music

    results = []

    # 1. 语音情感识别
    emotion_result = analyze_voice_emotion(audio_path)
    results.append({
        "module": "voice_emotion",
        "status": emotion_result.get("status", "error"),
        "summary": emotion_result.get("note", "语音情感分析未返回结果。"),
        "detail": emotion_result,
    })

    if emotion_result.get("status") != "success":
        return results

    stress = emotion_result.get("stress_level", "unknown")
    emotion = emotion_result.get("emotion", "未知")

    # 2. 如果压力偏高，触发呼吸训练干预
    if stress in ("high", "moderate"):
        breathing_result = generate_breathing_guidance(
            stress_level=stress,
            emotion=emotion,
        )
        results.append({
            "module": "breathing_intervention",
            "status": "success",
            "summary": f"已触发呼吸训练干预：{breathing_result['name']}，"
                       f"建议练习 {breathing_result['duration']}。",
            "detail": breathing_result,
        })

    # 3. 根据情感推荐音乐疗法
    music_result = recommend_music(
        emotion=emotion,
        stress_level=stress,
    )
    results.append({
        "module": "music_intervention",
        "status": "success",
        "summary": f"音乐推荐：{music_result['playlist_name']}，"
                   f"风格：{music_result['style']}。",
        "detail": music_result,
    })

    return results


def _run_deep_analysis(algo_id: str) -> dict[str, object]:
    """
    调用深度学习模型进行算法分析。
    根据算法 ID 分发到对应的真实分析函数。
    """
    try:
        if algo_id == "ecg":
            return analyze_ecg()
        elif algo_id == "pcg":
            return analyze_pcg()
        elif algo_id == "hrv":
            return analyze_hrv()
        elif algo_id == "cnn":
            return analyze_cnn()
        elif algo_id == "fusion":
            return analyze_fusion()
        else:
            return {"module": algo_id, "status": "unknown", "summary": f"未知算法: {algo_id}"}
    except Exception as e:
        return {
            "module": algo_id,
            "status": "error",
            "summary": f"算法调用失败: {e}",
            "risk_level": "unknown",
            "meta": {"placeholder_only": True},
        }


def _format_deep_analysis_for_llm(module: str, detail: dict) -> list[str]:
    """
    将深度学习分析结果格式化为 LLM 可理解的文本行。
    关键功能：把模型特征和分类结果注入 LLM 提示词。
    支持两种模式：
      - 有 deep_analysis（真实/演示推理）：输出详细特征和分类
      - 无 deep_analysis（mock/placeholder）：输出基础指标摘要
    """
    lines = []

    if module in ("ecg",):
        risk_map = {"low": "低风险", "moderate": "中等风险", "high": "高风险"}
        risk = detail.get("risk_level", "unknown")
        status = detail.get("status", "unknown")

        if "deep_analysis" in detail:
            # 有深度分析结果（真实推理或演示模式）
            da = detail["deep_analysis"]
            feat = da.get("features_summary", {})
            top = da.get("top_classes", [])

            # 特征信息
            if feat:
                if isinstance(feat.get('norm'), (int, float)):
                    lines.append(f"  特征维度: {feat.get('dim', 'N/A')}，"
                                 f"L2范数: {feat['norm']:.3f}")
                else:
                    lines.append(f"  特征维度: {feat.get('dim', 'N/A')}")
                lines.append(f"  特征统计: 均值 {feat.get('mean', 'N/A')}, "
                             f"标准差 {feat.get('std', 'N/A')}")

            # 分类结果
            if top:
                top_str = "、".join([f"{t['label']}({t['probability']:.1%})" for t in top[:3]])
                lines.append(f"  模型分类 Top3: {top_str}")

            lines.append(f"  综合风险: {risk_map.get(risk, risk)}")
        else:
            # Mock/placeholder 模式：使用基础指标
            metrics = detail.get("metrics", {})
            if metrics:
                hr = metrics.get("heart_rate_bpm", "N/A")
                rhythm = metrics.get("rhythm", "N/A")
                quality = metrics.get("signal_quality", "N/A")
                lines.append(f"  心率: {hr} bpm, 节律: {rhythm}, 信号质量: {quality}")
            lines.append(f"  分析模式: {status} (无深度模型结果)")
            lines.append(f"  综合风险: {risk_map.get(risk, risk)}")

    elif module in ("pcg",):
        risk_map = {"low": "低风险", "moderate": "中等风险", "high": "高风险"}
        status = detail.get("status", "unknown")

        if "deep_analysis" in detail:
            da = detail["deep_analysis"]
            feat = da.get("features_summary", {})
            pred = da.get("predictions", {})
            risk = detail.get("risk_assessment", {})

            if feat:
                if isinstance(feat.get('norm'), (int, float)):
                    lines.append(f"  特征维度: {feat.get('dim', 'N/A')}，"
                                 f"L2范数: {feat['norm']:.3f}")
                else:
                    lines.append(f"  特征维度: {feat.get('dim', 'N/A')}")

            if pred:
                for label, prob in pred.items():
                    lines.append(f"  {label}: {prob:.1%}")

            is_abnormal = risk.get("is_abnormal", False)
            score = risk.get("abnormality_score", 0)
            lines.append(f"  异常判定: {'是' if is_abnormal else '否'} (异常得分: {score:.1%})")
            lines.append(f"  听诊位置: {risk.get('auscultation_location', 'N/A')}")
        else:
            # Mock/placeholder 模式
            metrics = detail.get("metrics", {})
            if metrics:
                pattern = metrics.get("heart_sound_pattern", "N/A")
                murmur = metrics.get("murmur_risk", "N/A")
                lines.append(f"  心音模式: {pattern}, 杂音风险: {murmur}")
            lines.append(f"  分析模式: {status} (无深度模型结果)")
            lines.append(f"  综合风险: {risk_map.get(detail.get('risk_level', 'unknown'), detail.get('risk_level', 'unknown'))}")

    elif module == "cnn" and "prediction" in detail:
        pred = detail["prediction"]
        label = pred.get("label", "unknown")
        conf = pred.get("confidence", 0)
        probs = pred.get("probabilities", {})
        lines.append(f"  应激评估: {label} (置信度 {conf:.0%})")
        if probs:
            prob_str = "、".join([f"{k} {v:.0%}" for k, v in probs.items()])
            lines.append(f"  概率分布: {prob_str}")

    elif module == "fusion" or module == "transformer" and "prediction" in detail:
        pred = detail["prediction"]
        score = pred.get("health_score", 0)
        label = pred.get("fusion_label", "unknown")
        conf = pred.get("confidence", 0)
        label_map = {"healthy": "健康", "attention": "需关注", "warning": "警告", "critical": "危险"}
        lines.append(f"  综合健康评分: {score}/100 ({label_map.get(label, label)})")
        lines.append(f"  ECG 贡献: {pred.get('ecg_contribution', 'N/A')}分, "
                     f"HRV 贡献: {pred.get('hrv_contribution', 'N/A')}分, "
                     f"PCG 贡献: {pred.get('pcg_contribution', 'N/A')}分")
        lines.append(f"  融合特征维度: {pred.get('fused_feature_dim', 'N/A')}")
        meta = detail.get("meta", {})
        if meta.get("alignment_layer"):
            lines.append(f"  对齐层状态: {meta['alignment_layer']}")

    return lines


def _compute_numeric_scores(algorithm_results: list[dict]) -> dict[str, object]:
    """
    从算法分析结果中提取数值型 HRV 和压力分数。

    核心规则：
      - HRV 分数仅由用户上传的真实信号数据（ECG/PCG/HRV/fusion）决定，
        普通文字消息和语音消息不改变 HRV。
      - 压力分数由信号数据驱动；语音情感仅在极端（high）情况下微调。
      - 无真实数据时返回 None 表示"不改变当前值"。

    返回 {"hrv": int|None, "stress": int|None, "source": str}
    """
    # 分离真实数据来源 vs 关键词触发/语音分析
    file_results = [r for r in algorithm_results if r.get("data_source") == "file"]
    voice_results = [r for r in algorithm_results if r.get("module") == "voice_emotion"]

    # ======== HRV：仅从真实信号数据计算 ========
    hrv_score = None

    # 优先使用多模态融合结果（来自文件数据）
    for res in file_results:
        module = res.get("module", "")
        if module in ("transformer", "fusion") and "prediction" in res:
            pred = res["prediction"]
            if "health_score" in pred:
                hrv_score = int(round(pred["health_score"]))
                break

    if hrv_score is None:
        # 从单项 HRV 信号数据计算
        for res in file_results:
            module = res.get("module", "")
            if module == "hrv":
                metrics = res.get("metrics", {})
                sdnn = float(metrics.get("sdnn_ms", 42))
                rmssd = float(metrics.get("rmssd_ms", 35))
                sdnn_pct = min(1.0, sdnn / 80) * 100
                rmssd_pct = min(1.0, rmssd / 50) * 100
                hrv_score = int(round(0.5 * sdnn_pct + 0.5 * rmssd_pct))
                break

    # HRV 仍然为 None → 无真实数据，不改变
    # （文字消息、语音消息、关键词触发的 mock 分析均不会设置 HRV）

    # ======== Stress：由信号数据驱动 + 语音情感极端微调 ========
    stress_score = None

    # 从文件数据中提取压力分数
    for res in file_results:
        module = res.get("module", "")

        # fusion 结果
        if module in ("transformer", "fusion") and "prediction" in res:
            health = int(round(res["prediction"].get("health_score", 50)))
            stress_score = max(0, min(100, 100 - health))
            continue

        # CNN 应激评估
        if module == "cnn" and "prediction" in res:
            label = res["prediction"].get("label", "")
            if "high" in label:
                stress_score = max(stress_score or 0, 80)
            elif "moderate" in label:
                stress_score = max(stress_score or 0, 55)
            else:
                stress_score = max(stress_score or 0, 20)

        # HRV 分析中的 stress_level
        if module == "hrv":
            metrics = res.get("metrics", {})
            sl = metrics.get("stress_level", "moderate")
            rmap = {"high": 75, "moderate": 50, "low": 25}
            stress_score = max(stress_score or 0, rmap.get(sl, 40))

        # ECG 风险等级
        if module == "ecg":
            risk = res.get("risk_level", "low")
            rmap = {"low": 10, "moderate": 45, "high": 80}
            stress_score = max(stress_score or 0, rmap.get(risk, 30))

        # PCG 风险等级
        if module == "pcg":
            risk = res.get("risk_level", "low")
            rmap = {"low": 10, "moderate": 45, "high": 80}
            stress_score = max(stress_score or 0, rmap.get(risk, 30))

    # 语音情感：仅在极端（high）压力时才影响 stress，且不改变 HRV
    if voice_results:
        for res in voice_results:
            detail = res.get("detail") if isinstance(res.get("detail"), dict) else {}
            sl = detail.get("stress_level", "unknown")
            if sl == "high":
                # 极端情绪波动：压力上限设为 80，但不覆盖文件数据中更高的值
                stress_score = max(stress_score or 0, 80)
            # moderate/low 语音情感不改变压力分数

    # 如果完全没有数据来源，保持 None（不改变当前值）
    if hrv_score is None and stress_score is None:
        source = "unchanged"
    else:
        # 有数据时，互补缺失值
        if hrv_score is not None and stress_score is None:
            stress_score = max(0, min(100, 100 - hrv_score))
        elif stress_score is not None and hrv_score is None:
            hrv_score = max(0, min(100, 100 - stress_score))

        # 确保 HRV + STRESS 不超过 100
        if hrv_score + stress_score > 100:
            total = hrv_score + stress_score
            hrv_score = int(hrv_score / total * 100)
            stress_score = 100 - hrv_score

        source_parts = [r.get("module", "") for r in file_results if r.get("module")]
        if voice_results and any(
            r.get("detail", {}).get("stress_level") == "high" for r in voice_results
        ):
            source_parts.append("voice_emotion(extreme)")
        source = "+".join(source_parts) if source_parts else "default"

    return {"hrv": hrv_score, "stress": stress_score, "source": source}


def get_agent_response(
    user_message: str,
    session_id: str = "default",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    *,
    audio_path: str | None = None,
    uploaded_file: dict | None = None,
    model_client: LLMClient | None = None,
    memory_store: MemoryStore | None = None,
) -> dict[str, Any]:
    client = model_client or default_model_client
    store = memory_store or default_memory_store

    # 1. 处理上传文件（在保存用户消息之前，因为可能修改 user_message）
    file_algorithm_results = []
    if uploaded_file and uploaded_file.get("status") == "success":
        category = uploaded_file.get("category")
        if category == "report":
            # 体检单 OCR：调用 Qwen-VL 提取文字
            try:
                images = uploaded_file["content"]
                ocr_text = ""
                if isinstance(images, list):
                    for idx, img_b64 in enumerate(images):
                        page_text = client.vision_chat(
                            image_base64=img_b64,
                            prompt="请识别并提取这张体检报告中的所有文字内容、数值指标和异常标记。",
                        )
                        ocr_text += f"\n--- 第{idx+1}页 ---\n{page_text}"
                else:
                    ocr_text = client.vision_chat(image_base64=images)

                user_message = (
                    f"用户上传了体检报告（{uploaded_file.get('filename', '未知文件')}），"
                    f"请分析以下内容：\n\n{ocr_text}"
                )
            except Exception as e:
                user_message += f"\n[体检单 OCR 解析失败: {e}]"

        elif category == "signal":
            # 信号数据文件：调用对应的算法分析函数
            signal_type = uploaded_file.get("signal_type")
            signal_content = uploaded_file.get("content", {})

            try:
                if signal_type == "ecg":
                    result = analyze_ecg(signal_data=signal_content.get("data"))
                    result["data_source"] = "file"  # 标记为真实数据来源
                    file_algorithm_results.append(result)
                    user_message += "\n[用户上传了 ECG 心电信号数据文件]"
                elif signal_type == "pcg":
                    result = analyze_pcg(audio_data=signal_content.get("data"))
                    result["data_source"] = "file"
                    file_algorithm_results.append(result)
                    user_message += "\n[用户上传了 PCG 心音信号文件]"
                elif signal_type == "hrv":
                    result = analyze_hrv(rr_intervals=signal_content.get("data"))
                    result["data_source"] = "file"
                    file_algorithm_results.append(result)
                    user_message += "\n[用户上传了 HRV 心率变异数据]"
                else:
                    user_message += (
                        f"\n[用户上传了信号数据文件 {uploaded_file.get('filename', '未知')}，"
                        f"数据形状: {signal_content.get('shape', 'unknown')}]"
                    )
            except Exception as e:
                user_message += f"\n[信号数据分析失败: {e}]"

    # 2. 获取并保存用户消息
    conversation = store.get_session(session_id.strip() or "default")
    conversation.add_message(role="user", content=user_message)

    # 3. 算法分析逻辑（关键词匹配，仅对文本消息生效）
    algorithm_results = list(file_algorithm_results)
    lower_msg = user_message.lower()
    for algo_id, keywords in ALGORITHM_KEYWORDS.items():
        if any(kw in lower_msg for kw in keywords):
            result = _run_deep_analysis(algo_id)
            algorithm_results.append(result)

    # 3. 语音情感分析（仅在提供音频文件时触发）
    if audio_path:
        try:
            voice_results = _run_voice_emotion_analysis(audio_path)
            if isinstance(voice_results, list):
                algorithm_results.extend(voice_results)
        except Exception as e:
            algorithm_results.append({
                "module": "voice_emotion",
                "status": "error",
                "summary": f"语音情感分析失败: {e}",
            })

    # 4. 构造增强提示词（包含模型特征和分类结果）
    model_input = user_message
    if algorithm_results:
        algo_lines = []
        for res in algorithm_results:
            module = res.get("module", "unknown")
            summary = res.get("summary", "")
            algo_lines.append(f"- {module}: {summary}")

            # 将深度学习分析的详细结果注入 LLM 提示词
            detail = res.get("detail")
            if isinstance(detail, dict) and module in ("breathing_intervention", "music_intervention"):
                if module == "breathing_intervention":
                    algo_lines.append(f"  呼吸方案：{' → '.join(detail.get('steps', []))}，时长：{detail.get('duration', '')}")
                    algo_lines.append(f"  效果说明：{detail.get('effect', '')}")
                elif module == "music_intervention":
                    algo_lines.append(f"  音乐风格：{detail.get('style', '')}，节奏：{detail.get('tempo', '')}")
                    algo_lines.append(f"  推荐曲目：{'、'.join(detail.get('recommendations', [])[:3])}")
                    algo_lines.append(f"  收听指南：{detail.get('listening_guide', '')}")
            elif isinstance(detail, dict) and module == "voice_emotion":
                scores = detail.get("scores", {})
                top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                score_str = "、".join([f"{EMOTION_CN_MAP.get(k, k)} {v:.1%}" for k, v in top3])
                algo_lines.append(f"  情感分布 Top3：{score_str}")
            else:
                # 深度学习模型结果（ECG/PCG/CNN/Fusion）
                deep_lines = _format_deep_analysis_for_llm(module, detail if isinstance(detail, dict) else res)
                algo_lines.extend(deep_lines)

        algo_summary = "\n".join(algo_lines)
        model_input += f"\n\n[系统算法分析结果]:\n{algo_summary}"

    try:
        history = conversation.get_messages(include_system=False)
        if history and history[-1]["role"] == "user":
            history = history[:-1]

        response = client.chat(
            user_message=model_input,
            history=history,
            system_prompt=system_prompt,
        )
        reply_text = response.text
        conversation.add_message(role="assistant", content=reply_text)

    except Exception as e:
        reply_text = f"抱歉，系统调用模型时出现错误: {str(e)}"
        conversation.add_message(role="assistant", content=reply_text)

    # 算法直出数值型分数
    health_scores = _compute_numeric_scores(algorithm_results)

    return {
        "reply": reply_text,
        "algorithms": algorithm_results,
        "health_scores": health_scores,
        "session_id": session_id,
    }

# 情感中文名映射（供 core.py 内部使用）
EMOTION_CN_MAP = {
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


def get_session_history(
    session_id: str = "default",
    *,
    include_system: bool = True,
    memory_store: MemoryStore | None = None,
) -> list[dict[str, str]]:
    store = memory_store or default_memory_store
    conversation = store.get_session(session_id.strip() or "default")
    return conversation.get_messages(include_system=include_system)

def clear_session_history(
    session_id: str = "default",
    *,
    keep_system_message: bool = True,
    memory_store: MemoryStore | None = None,
) -> None:
    store = memory_store or default_memory_store
    store.clear_session(
        session_id.strip() or "default",
        keep_system_message=keep_system_message,
    )

def _safe_algorithm_call(function: Any, **kwargs: Any) -> dict[str, object]:
    try:
        result = function(**kwargs)
    except Exception as exc:
        module_name = getattr(function, "__name__", "algorithm")
        return {
            "module": module_name,
            "status": "error",
            "summary": f"算法调用失败: {exc}",
            "risk_level": "unknown",
            "meta": {"placeholder_only": True},
        }
    return dict(result)
