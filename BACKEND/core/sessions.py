import threading
import time
from collections import OrderedDict


class SessionStore:
    def __init__(self, ttl_s: int = 7200, max_sessions: int = 500, max_turns: int = 5, clock=time.monotonic):
        self.ttl_s, self.max_sessions, self.max_turns, self._clock = ttl_s, max_sessions, max_turns, clock
        self._data = OrderedDict()          
        self._lock = threading.Lock()

    def _purge(self):
        now = self._clock()
        for sid in [s for s, v in self._data.items() if now - v["ts"] > self.ttl_s]:
            del self._data[sid]
        while len(self._data) > self.max_sessions:
            self._data.popitem(last=False)

    def add_turn(self, sid: str, user: str, assistant: str):
        with self._lock:
            s = self._data.setdefault(sid, {"turns": [], "ts": self._clock()})
            s["turns"].append((user, assistant))
            s["turns"] = s["turns"][-self.max_turns:]
            s["ts"] = self._clock()
            self._data.move_to_end(sid)
            self._purge()               

    def history_text(self, sid: str, max_chars: int = 1200) -> str:
        with self._lock:
            self._purge()
            s = self._data.get(sid)
            if not s:
                return ""
            text = "\n".join(f"Utilisateur : {u[:400]}\nAssistant : {a[:500]}" for u, a in s["turns"])
        return text[-max_chars:]

    def clear(self, sid: str) -> bool:
        with self._lock:
            return self._data.pop(sid, None) is not None

    def __len__(self):
        with self._lock:
            return len(self._data)
