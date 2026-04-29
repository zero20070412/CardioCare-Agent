import gradio as gr

def create_chart_panel():
    with gr.Row():
        with gr.Column(variant="panel", scale=1):
            gr.Markdown("### HRV 活力状态")
            hrv_slider = gr.Slider(0, 100, value=0, label="HRV 指标", interactive=False)
            stress_slider = gr.Slider(0, 100, value=0, label="应激压力", interactive=False)

        with gr.Column(variant="panel", scale=1):
            gr.Markdown("### 健康建议")
            health_tips = gr.Textbox(
                value="等待对话分析...",
                show_label=False,
                interactive=False,
                lines=5.38
            )
                
    return None, hrv_slider, stress_slider, health_tips