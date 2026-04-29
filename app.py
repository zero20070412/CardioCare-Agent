import gradio as gr
import os
import uuid
import time
from frontend.sidebar import create_sidebar
from frontend.chart_panel import create_chart_panel
from frontend.chat_panel import create_chat_panel
from agent.core import get_agent_response, clear_session_history
from agent.prompts import CARDIO_ASSISTANT_PROMPT
from agent.history_manager import save_session, get_history_choices, get_session_history

# 初始化全局会话 ID
SESSION_ID = str(uuid.uuid4())

gemini_ultimate_css = """
.gradio-container { background-color: #ffffff !important; }
.gr-panel, .gr-block { border: none !important; box-shadow: none !important; }

#gemini-capsule-row {
    background-color: #FFFFFF !important;
    border: 1px solid #E8EAED !important;
    border-radius: 32px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    padding: 6px 16px !important;
    align-items: flex-end !important;
    gap: 4px !important;
}

#gemini-input-box textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    font-size: 16px !important;
    padding-left: 4px !important;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
}
#gemini-input-box textarea:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
#gemini-input-box .container {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

#gemini-audio-box {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    margin-left: -8px !important;
}
#gemini-audio-box button[aria-label="Clear"] {
    display: none !important;
}

#toggle-icon, #gemini-send-btn {
    background: transparent !important;
    border: none !important;
    min-width: 44px !important;
    max-width: 44px !important;
    height: 44px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-bottom: 2px !important;
}

#toggle-icon img {
    width: 28px !important;
    height: 28px !important;
    object-fit: contain !important;
    transform: scale(1.3) !important;
    transform-origin: center !important;
}

#gemini-send-btn {
    background-color: #F1F3F4 !important;
    border-radius: 50% !important;
    color: #202124 !important;
    font-size: 18px !important;
}

.thinking-container {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 10px 0;
}

.thinking-dot {
    width: 8px;
    height: 8px;
    background-color: #4285F4;
    border-radius: 50%;
    opacity: 0.4;
    animation: thinking-bounce 1.4s infinite ease-in-out both;
}

.thinking-dot:nth-child(1) { animation-delay: -0.32s; }
.thinking-dot:nth-child(2) { animation-delay: -0.16s; }

@keyframes thinking-bounce {
    0%, 80%, 100% { transform: scale(0.8); opacity: 0.4; }
    40% { transform: scale(1.2); opacity: 1; }
}
"""


def handle_file_upload(file):
    """处理文件上传：解析文件类型并显示状态"""
    if file is None:
        return "未选择文件", None

    file_path = file if isinstance(file, str) else str(file)
    from algorithms.file_parser.parser import parse_uploaded_file

    result = parse_uploaded_file(file_path)

    if result.get("status") == "success":
        category_map = {"report": "体检报告", "signal": "信号数据"}
        signal_map = {"ecg": "ECG 心电", "pcg": "PCG 心音", "hrv": "HRV 心率变异"}
        cat = category_map.get(result["category"], result.get("category", ""))
        sig = signal_map.get(result.get("signal_type"), "")

        msg = f"已解析: {result['filename']} ({cat}"
        if sig:
            msg += f", {sig}"
        msg += ")，发送消息后将自动注入分析结果。"
        return msg, result
    else:
        return f"解析失败: {result.get('message', '未知错误')}", None


def _extract_text_from_multimedia(content):
    """从多媒体消息 content 中提取纯文本（兼容 Gradio 6.x 格式）"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            # Gradio 6.x NormalizedMessageContent: {"type": "text", "text": "..."}
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                # 跳过 file/component 类型的 content item
            elif isinstance(item, str):
                text_parts.append(item)
            # 兼容旧 Gradio 5.x tuple 格式: ("text", "...")
            elif isinstance(item, tuple) and len(item) == 2 and item[0] == "text":
                text_parts.append(str(item[1]))
        return " ".join(text_parts).strip() if text_parts else ""
    return str(content) if content else ""


def _parse_health_tip(reply_text: str) -> tuple[str, str]:
    """
    从 LLM 回复中解析健康建议 TIP。
    返回 (tip, clean_text)。从展示文本中移除 [HEALTH_METRICS] 标签块。
    """
    import re

    pattern = r'\[HEALTH_METRICS\]\s*.*?\s*TIP:\s*(.+?)\s*\[/HEALTH_METRICS\]'
    match = re.search(pattern, reply_text, re.DOTALL)

    if match:
        tip = match.group(1).strip().rstrip('。')
        clean_text = reply_text[:match.start()].rstrip() + "\n\n" + reply_text[match.end():].lstrip()
        return tip, clean_text

    return "", reply_text


def handle_chat(user_text, user_audio, input_mode, chat_history, parsed_data):
    global SESSION_ID

    transcribed_text = None
    audio_file_path = None

    if input_mode == "text":
        if not isinstance(user_text, str) or user_text.strip() == "":
            yield "", None, chat_history, gr.update(), gr.update(), gr.update(), gr.update()
            return
        user_message = user_text.strip()
    else:
        if not user_audio:
            yield "", None, chat_history, gr.update(), gr.update(), gr.update(), gr.update()
            return

        audio_file_path = user_audio if isinstance(user_audio, str) else str(user_audio)

        # ASR 语音转文字
        try:
            from algorithms.asr.asr import transcribe_audio
            asr_result = transcribe_audio(audio_file_path)
            if asr_result.get("status") == "success":
                transcribed_text = asr_result["text"]
                user_message = transcribed_text
            else:
                user_message = "[接收到语音文件]"
        except Exception:
            user_message = "[接收到语音文件]"

    # 构建用户消息（支持多媒体格式，Gradio 6.x）
    if input_mode == "audio" and audio_file_path:
        display_text = transcribed_text if transcribed_text else "[语音消息]"
        user_content = [
            display_text,
            {"path": audio_file_path, "meta": {"_type": "gradio.FileData"}},
        ]
        chat_history.append({"role": "user", "content": user_content})
    else:
        chat_history.append({"role": "user", "content": user_message})

    thinking_html = '<div class="thinking-container"><div class="thinking-dot"></div><div class="thinking-dot"></div><div class="thinking-dot"></div></div>'
    chat_history.append({"role": "assistant", "content": thinking_html})
    yield "", None, chat_history, gr.update(), gr.update(), gr.update(), gr.update()

    try:
        from agent.core import default_memory_store, clear_session_history
        conv = default_memory_store.get_session(SESSION_ID)

        valid_frontend_msgs = [
            m for m in chat_history[:-2]
            if "thinking-container" not in _extract_text_from_multimedia(m.get("content", ""))
        ]
        mem_msgs = conv.get_messages(include_system=False)

        if len(mem_msgs) < len(valid_frontend_msgs):
            missing_msgs = valid_frontend_msgs[len(mem_msgs):]
            for msg in missing_msgs:
                text_content = _extract_text_from_multimedia(msg.get("content", ""))
                conv.add_message(role=msg["role"], content=text_content)
        elif len(mem_msgs) > len(valid_frontend_msgs):
            clear_session_history(SESSION_ID)
            for msg in valid_frontend_msgs:
                text_content = _extract_text_from_multimedia(msg.get("content", ""))
                conv.add_message(role=msg["role"], content=text_content)

        response = get_agent_response(
            user_message=user_message,
            session_id=SESSION_ID,
            system_prompt=CARDIO_ASSISTANT_PROMPT,
            audio_path=audio_file_path,
            uploaded_file=parsed_data,
        )
        full_reply = response.get("reply", "...")

        # 从 LLM 回复中解析健康建议
        health_tip, display_reply = _parse_health_tip(full_reply)
        if not health_tip:
            health_tip = "持续关注心血管健康，保持良好的生活习惯。"

        # 流式输出文字（文字动画）
        chat_history[-1]["content"] = ""
        current_text = ""
        for char in display_reply:
            current_text += char
            chat_history[-1]["content"] = current_text
            yield "", None, chat_history, gr.update(), gr.update(), gr.update(), gr.update()
            time.sleep(0.01)

        # HRV/压力面板暂固定为 0，待后续算法开发
        save_session(SESSION_ID, chat_history)
        history_choices = get_history_choices()
        yield (
            "", None, chat_history,
            gr.update(choices=history_choices),
            gr.update(value=0),
            gr.update(value=0),
            gr.update(value=health_tip),
        )
    except Exception as e:
        chat_history[-1]["content"] = f"系统错误: {str(e)}"
        yield "", None, chat_history, gr.update(), gr.update(), gr.update(), gr.update()


def toggle_input_mode(current_mode, mic_path, kb_path):
    if current_mode == "text":
        return "audio", gr.update(visible=False), gr.update(visible=True), gr.update(icon=kb_path)
    else:
        return "text", gr.update(visible=True), gr.update(visible=False), gr.update(icon=mic_path)


def start_new_chat():
    global SESSION_ID
    SESSION_ID = str(uuid.uuid4())
    clear_session_history(SESSION_ID)
    gr.Info("已开启新对话！")
    return [], gr.update(value=None), gr.update(value=0), gr.update(value=0), gr.update(value="等待对话分析...")


def load_history(selected_session_id):
    global SESSION_ID
    if selected_session_id:
        SESSION_ID = selected_session_id
        history = get_session_history(selected_session_id)

        from agent.core import clear_session_history, default_memory_store
        clear_session_history(selected_session_id)
        conv = default_memory_store.get_session(selected_session_id)
        for msg in history:
            text_content = _extract_text_from_multimedia(msg.get("content", ""))
            if "thinking-container" not in text_content:
                conv.add_message(role=msg["role"], content=text_content)
        return history
    return []


# Gradio 界面布局

with gr.Blocks(title="CardioBot") as demo:
    gr.Markdown("# CardioCare 健康助手")

    input_mode = gr.State("text")

    with gr.Row():
        with gr.Column(scale=2):
            up_btn, parse_status, parsed_data, new_btn, history_dropdown = create_sidebar()
        with gr.Column(scale=8):
            chat_box, input_box, audio_box, toggle_b, send_b, mic_path, kb_path = create_chat_panel()
            _, hrv_s, stress_s, tips_t = create_chart_panel()

    # 上传文件事件
    up_btn.change(
        fn=handle_file_upload,
        inputs=[up_btn],
        outputs=[parse_status, parsed_data],
    )

    toggle_b.click(
        fn=toggle_input_mode,
        inputs=[input_mode, gr.State(mic_path), gr.State(kb_path)],
        outputs=[input_mode, input_box, audio_box, toggle_b]
    )

    send_b.click(
        fn=handle_chat,
        inputs=[input_box, audio_box, input_mode, chat_box, parsed_data],
        outputs=[input_box, audio_box, chat_box, history_dropdown, hrv_s, stress_s, tips_t],
        show_progress="hidden"
    )

    input_box.submit(
        fn=handle_chat,
        inputs=[input_box, audio_box, input_mode, chat_box, parsed_data],
        outputs=[input_box, audio_box, chat_box, history_dropdown, hrv_s, stress_s, tips_t],
        show_progress="hidden"
    )

    new_btn.click(
        fn=start_new_chat,
        inputs=[],
        outputs=[chat_box, history_dropdown, hrv_s, stress_s, tips_t]
    )

    history_dropdown.change(
        fn=load_history,
        inputs=[history_dropdown],
        outputs=[chat_box]
    )

if __name__ == "__main__":
    root_path = os.path.dirname(os.path.abspath(__file__))
    demo.launch(css=gemini_ultimate_css, allowed_paths=[root_path])
