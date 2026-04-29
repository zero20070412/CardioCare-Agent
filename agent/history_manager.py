import json
import os

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")

def load_all_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_all_history(data):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_title(chat_history):
    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, dict):
                content = content.get("text", "")
            elif isinstance(content, list):
                # 多媒体消息：提取 text 部分
                # Gradio 6.x 格式: [{"type": "text", "text": "..."}, ...]
                # Gradio 5.x 格式: [("text", "..."), ("audio", ...)]
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(str(item.get("text", "")))
                    elif isinstance(item, tuple) and len(item) == 2 and item[0] == "text":
                        text_parts.append(str(item[1]))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = " ".join(text_parts).strip() if text_parts else ""

            content = str(content)
            content = content.replace("[接收到语音文件]", "").strip()
            content = content.replace("[语音消息]", "").strip()

            if content:
                return content[:12] + "..." if len(content) > 12 else content

    return "新对话"

def save_session(session_id, chat_history):
    if not chat_history:
        return
    history_data = load_all_history()
    title = generate_title(chat_history)
    history_data[session_id] = {
        "title": title,
        "history": chat_history
    }
    save_all_history(history_data)

def get_history_choices():
    data = load_all_history()
    choices = [(info["title"], sid) for sid, info in data.items()]
    return choices[::-1] 

def get_session_history(session_id):
    data = load_all_history()
    return data.get(session_id, {}).get("history", [])