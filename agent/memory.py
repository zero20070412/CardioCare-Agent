from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from uuid import uuid4

from utils.config import settings


VALID_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """Store and trim the message history for a single conversation."""

    def __init__(self, max_rounds: int | None = None) -> None:
        self.max_rounds = max_rounds if max_rounds is not None else settings.max_history_rounds
        self._messages: list[ChatMessage] = []

    def set_system_message(self, content: str) -> None:
        """Insert or replace the leading system prompt for the conversation."""
        normalized = self._normalize_content(content)
        system_message = ChatMessage(role="system", content=normalized)

        if self._messages and self._messages[0].role == "system":
            self._messages[0] = system_message
            return

        self._messages.insert(0, system_message)

    def add_message(self, role: str, content: str) -> ChatMessage:
        normalized_role = role.strip().lower()
        if normalized_role not in VALID_ROLES:
            raise ValueError(f"Unsupported role: {role}")

        normalized_content = self._normalize_content(content)
        message = ChatMessage(role=normalized_role, content=normalized_content)
        self._messages.append(message)
        self._trim_history()
        return message

    def add_user_message(self, content: str) -> ChatMessage:
        return self.add_message("user", content)

    def add_assistant_message(self, content: str) -> ChatMessage:
        return self.add_message("assistant", content)

    def add_messages(self, messages: Iterable[dict[str, str] | ChatMessage]) -> None:
        for message in messages:
            if isinstance(message, ChatMessage):
                self.add_message(message.role, message.content)
            else:
                self.add_message(message["role"], message["content"])

    def get_messages(self, *, include_system: bool = True) -> list[dict[str, str]]:
        if include_system:
            return [message.to_dict() for message in self._messages]

        return [
            message.to_dict()
            for message in self._messages
            if message.role != "system"
        ]

    def get_recent_messages(
        self,
        *,
        limit: int | None = None,
        include_system: bool = True,
    ) -> list[dict[str, str]]:
        messages = self.get_messages(include_system=include_system)
        if limit is None or limit <= 0:
            return messages
        return messages[-limit:]

    def clear(self, *, keep_system_message: bool = False) -> None:
        if keep_system_message and self._messages and self._messages[0].role == "system":
            self._messages = [self._messages[0]]
            return

        self._messages = []

    def is_empty(self) -> bool:
        non_system_messages = [
            message for message in self._messages if message.role != "system"
        ]
        return len(non_system_messages) == 0

    def message_count(self, *, include_system: bool = True) -> int:
        if include_system:
            return len(self._messages)
        return sum(1 for message in self._messages if message.role != "system")

    def _trim_history(self) -> None:
        if self.max_rounds <= 0:
            return

        system_message = self._messages[0] if self._messages and self._messages[0].role == "system" else None
        non_system_messages = [
            message for message in self._messages if message.role != "system"
        ]

        max_messages = self.max_rounds * 2
        if len(non_system_messages) <= max_messages:
            return

        trimmed_messages = non_system_messages[-max_messages:]
        self._messages = ([system_message] if system_message else []) + trimmed_messages

    @staticmethod
    def _normalize_content(content: str) -> str:
        normalized = content.strip()
        if not normalized:
            raise ValueError("Message content cannot be empty.")
        return normalized


class MemoryStore:
    """Manage multiple conversations keyed by a session id."""

    def __init__(self, max_rounds: int | None = None) -> None:
        self.max_rounds = max_rounds if max_rounds is not None else settings.max_history_rounds
        self._sessions: dict[str, ConversationMemory] = {}

    def create_session(self, session_id: str | None = None) -> str:
        resolved_session_id = session_id or uuid4().hex
        self._sessions[resolved_session_id] = ConversationMemory(
            max_rounds=self.max_rounds
        )
        return resolved_session_id

    def get_session(self, session_id: str) -> ConversationMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(max_rounds=self.max_rounds)
        return self._sessions[session_id]

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def clear_session(self, session_id: str, *, keep_system_message: bool = False) -> None:
        session = self.get_session(session_id)
        session.clear(keep_system_message=keep_system_message)

    def clear_all(self) -> None:
        self._sessions = {}

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())


default_memory_store = MemoryStore()
