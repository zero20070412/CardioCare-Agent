import gradio as gr
import os

# 1. 确保在函数外部定义了这些路径
current_dir = os.path.dirname(os.path.abspath(__file__))
mic_icon_path = os.path.join(current_dir, "213麦克风.png")
kb_icon_path = os.path.join(current_dir, "键盘.webp")

def create_chat_panel():
    with gr.Column():
        chatbot = gr.Chatbot(label=None, show_label=False, height=500)

        with gr.Row(elem_id="gemini-capsule-row"):
            # 文本输入框
            msg_input = gr.Textbox(
                placeholder="发信息...",
                container=False,
                show_label=False,
                lines=1,
                max_lines=6,
                elem_id="gemini-input-box",
                scale=20
            )

            # 语音输入框
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                visible=False,
                elem_id="gemini-audio-box",
                scale=20,
                show_label=False,
                container=False,
            )

            # 2. 这里引用上面的 mic_icon_path 就不会报错了
            toggle_btn = gr.Button(
                value="",
                icon=mic_icon_path,
                elem_id="toggle-icon",
                visible=True
            )

            send_btn = gr.Button("➔", elem_id="gemini-send-btn", visible=True)

    # 3. 确保返回的变量名与 app.py 接收的一一对应
    return chatbot, msg_input, audio_input, toggle_btn, send_btn, mic_icon_path, kb_icon_path
