from collections import deque
from threading import Lock
from time import strftime


_entries = deque(maxlen=200)
_lock = Lock()


def clear_runtime_log():
    with _lock:
        _entries.clear()


def append_runtime_log(message):
    if not message:
        return
    with _lock:
        _entries.append(f"[{strftime('%H:%M:%S')}] {message}")


def get_runtime_log():
    with _lock:
        return "\n".join(_entries) or "実行待機中"
