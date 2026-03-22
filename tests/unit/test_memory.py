import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from edadvisor.generation.memory import SessionMemory


class TestSessionMemoryFallback:
    """Tests using in-process fallback (no Redis needed)."""

    def _mem(self):
        m = SessionMemory(max_turns=3)
        m._client = None   # force fallback
        return m

    def test_empty_session_returns_empty_list(self):
        mem = self._mem()
        assert mem.get_history("new-session") == []

    def test_add_turn_stores_both_messages(self):
        mem = self._mem()
        mem.add_turn("s1", "What visa do I need?", "You need a Student visa.")
        history = mem.get_history("s1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_multiple_turns_accumulate(self):
        mem = self._mem()
        mem.add_turn("s1", "Q1", "A1")
        mem.add_turn("s1", "Q2", "A2")
        assert len(mem.get_history("s1")) == 4

    def test_max_turns_respected(self):
        mem = self._mem()
        for i in range(10):
            mem.add_turn("s1", f"Q{i}", f"A{i}")
        history = mem.get_history("s1")
        # max_turns=3 → keep last 3*2=6 messages
        assert len(history) <= 6

    def test_clear_removes_session(self):
        mem = self._mem()
        mem.add_turn("s1", "Q", "A")
        mem.clear("s1")
        assert mem.get_history("s1") == []

    def test_sessions_isolated(self):
        mem = self._mem()
        mem.add_turn("session-a", "Q", "A")
        assert mem.get_history("session-b") == []

    def test_format_history_empty_returns_empty_string(self):
        mem = self._mem()
        assert mem.format_history_for_prompt("empty") == ""

    def test_format_history_has_role_labels(self):
        mem = self._mem()
        mem.add_turn("s1", "question", "answer")
        formatted = mem.format_history_for_prompt("s1")
        assert "Student" in formatted
        assert "EdAdvisor" in formatted
