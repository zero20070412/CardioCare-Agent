from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import os

from utils.config import settings

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# 定义类型别名
ChatMessage = Mapping[str, str]

class ModelClientError(RuntimeError):
    """当模型客户端无法完成请求时抛出。"""

@dataclass(frozen=True)
class ModelResponse:
    text: str
    model_name: str
    used_mock: bool

def build_messages(
    *,
    system_prompt: str | None = None,
    history: Sequence[Any] | None = None,
    user_message: str,
) -> list[dict[str, str]]:
 
    messages: list[dict[str, str]] = []

    # 1. 注入系统提示词
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 2. 转换历史对话（处理 Gradio 传入的 [[u, a], ...] 格式）
    for item in history or []:
        # 情况 A: 标准字典格式 {"role": "...", "content": "..."}
        if isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            # Gradio 多模态消息的 content 可能是 list，需强转
            if isinstance(role, list):
                role = str(role[0]) if role else ""
            if isinstance(content, list):
                # 多媒体消息：提取文本部分
                # Gradio 6.x 格式: [{"type": "text", "text": "..."}, {"type": "file", ...}]
                # Gradio 5.x 格式: [("text", "..."), ("audio", ...)]
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(str(item.get("text", "")))
                    elif isinstance(item, tuple) and len(item) == 2:
                        if item[0] == "text":
                            text_parts.append(str(item[1]))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = " ".join(text_parts).strip() if text_parts else ""
            role = str(role).strip()
            content = str(content).strip()
            if role in {"system", "user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        
        # 情况 B: Gradio 默认的列表或元组嵌套格式 [[user, bot]]
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            u_text, a_text = item
            if u_text:
                messages.append({"role": "user", "content": str(u_text).strip()})
            if a_text:
                messages.append({"role": "assistant", "content": str(a_text).strip()})

    # 3. 注入当前用户最新的一条消息
    messages.append({"role": "user", "content": user_message})
    return messages

class LLMClient:
    """针对阿里百炼等兼容 OpenAI 协议的 API 封装。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        use_mock: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        # 优先从 .env 加载配置，否则使用 settings 默认值
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.base_url = base_url if base_url is not None else settings.openai_base_url
        self.model_name = model_name if model_name is not None else settings.model_name
        self.use_mock = use_mock if use_mock is not None else settings.use_mock_model
        
        self.temperature = temperature if temperature is not None else settings.model_temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.model_max_tokens
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.model_timeout_seconds

        self._client: Any | None = None

    def chat(
        self,
        *,
        user_message: str,
        history: Sequence[Any] | None = None,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        """核心对话入口"""
        # 生成标准消息列表
        messages = build_messages(
            system_prompt=system_prompt,
            history=history,
            user_message=user_message,
        )

        # 判断是否进入模拟模式
        if self.use_mock:
            return self._mock_chat(user_message=user_message)

        return self._real_chat(messages=messages)

    def _real_chat(self, *, messages: Sequence[Mapping[str, str]]) -> ModelResponse:
        """执行正式 API 请求"""
        client = self._get_client()

        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=list(messages),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise ModelClientError(f"API 请求失败: {exc}") from exc

        text = self._extract_text(completion)
        return ModelResponse(
            text=text,
            model_name=self.model_name,
            used_mock=False,
        )

    def _mock_chat(self, *, user_message: str) -> ModelResponse:
        """本地模拟模式回复"""
        text = f"[MOCK] 收到：{user_message}。当前处于测试模式，请在 .env 中关闭 USE_MOCK_MODEL 以连接 API。"
        return ModelResponse(text=text, model_name=self.model_name, used_mock=True)

    def vision_chat(
        self,
        *,
        image_base64: str | list[str],
        prompt: str = "请识别并提取这张体检报告中的所有文字内容和数值指标。",
        system_prompt: str | None = None,
    ) -> str:
        """
        使用 Qwen-VL 进行图片理解（体检单 OCR）。
        通过 DashScope 兼容 OpenAI API 发送图片。

        Args:
            image_base64: 单张或多张图片的 base64 data URI（如 "data:image/png;base64,..."）
            prompt: 发送给视觉模型的提示词
            system_prompt: 系统提示词

        Returns:
            str: 模型返回的文字内容
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 构建多模态用户消息
        user_content = []
        if isinstance(image_base64, list):
            for img in image_base64:
                user_content.append({"type": "image_url", "image_url": {"url": img}})
        else:
            user_content.append({"type": "image_url", "image_url": {"url": image_base64}})
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        client = self._get_client()
        try:
            # 使用 qwen-vl-max 多模态模型
            vision_model = os.getenv("VISION_MODEL_NAME", "qwen-vl-max")
            completion = client.chat.completions.create(
                model=vision_model,
                messages=messages,
                max_tokens=2000,
            )
            return self._extract_text(completion)
        except Exception as exc:
            raise ModelClientError(f"视觉理解请求失败: {exc}") from exc

    def _get_client(self) -> Any:
        """懒加载初始化 OpenAI 客户端"""
        if self._client is not None:
            return self._client

        if OpenAI is None:
            raise ModelClientError("未检测到 openai 库，请执行 `pip install openai`。")

        if not self.api_key:
            raise ModelClientError("API_KEY 缺失，请检查 .env 文件。")

        # 初始化 DashScope 兼容配置
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )
        return self._client

    @staticmethod
    def _extract_text(completion: Any) -> str:
        """安全提取返回的文本内容"""
        try:
            content = completion.choices[0].message.content
            return content.strip() if content else ""
        except (AttributeError, IndexError, KeyError) as exc:
            raise ModelClientError(f"解析 API 响应失败: {exc}")

# 全局单例客户端
default_model_client = LLMClient()