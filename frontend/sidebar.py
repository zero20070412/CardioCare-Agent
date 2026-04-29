import gradio as gr
from agent.history_manager import get_history_choices


def create_sidebar():
    with gr.Column(variant="panel"):
        up_btn = gr.File(
            label="上传文件（体检单/信号数据）",
            file_types=[".pdf", ".jpg", ".png", ".csv", ".npy", ".wav"],
        )
        parse_status = gr.Textbox(
            label="解析状态",
            interactive=False,
            placeholder="支持: PDF/JPG/PNG 体检单, CSV/NPY/WAV 信号数据",
            lines=2,
            max_lines=4,
        )
        parsed_data = gr.State(None)
        new_btn = gr.Button("新对话", variant="primary")

        history_dropdown = gr.Dropdown(
            label="历史对话记录",
            choices=get_history_choices(),
            interactive=True,
            value=None,
        )

    return up_btn, parse_status, parsed_data, new_btn, history_dropdown
