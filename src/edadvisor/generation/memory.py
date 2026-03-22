from __future__ import annotations

import json

from loguru import logger

from edadvisor.config import settings


class SessionMemory:
    """
    Sliding-window conversation memory backed by Redis.
    Falls back to in-process dict if Redis is unavailable.

    Stores the last `max_turns` exchanges per session_id.
    Each exchange is {role: "user"|"assistant", content: str}.
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._client = None
        self._fallback: dict[str, list] = {}
        self._try_connect()

    def _try_connect(self) -> None:
        try:
            import redis
            r = redis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=2)
            r.ping()
            self._client = r
            logger.info("Redis connected for session memory")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}) — using in-process dict fallback")

    def get_history(self, session_id: str) -> list[dict]:
        if self._client:
            raw = self._client.get(f"session:{session_id}")
            return json.loads(raw) if raw else []
        return self._fallback.get(session_id, [])

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        history = self.get_history(session_id)
        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})

        # keep only the last max_turns exchanges
        history = history[-(self.max_turns * 2):]

        if self._client:
            self._client.setex(
                f"session:{session_id}",
                settings.session_ttl_seconds,
                json.dumps(history),
            )
        else:
            self._fallback[session_id] = history

    def clear(self, session_id: str) -> None:
        if self._client:
            self._client.delete(f"session:{session_id}")
        else:
            self._fallback.pop(session_id, None)

    def format_history_for_prompt(self, session_id: str) -> str:
        """Return history as a readable string to inject into the prompt."""
        history = self.get_history(session_id)
        if not history:
            return ""
        lines = []
        for turn in history:
            prefix = "Student" if turn["role"] == "user" else "EdAdvisor"
            lines.append(f"{prefix}: {turn['content']}")
        return "\n".join(lines)
