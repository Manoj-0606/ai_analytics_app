# app/cache.py
import time
from typing import Any

class SimpleTTLCache:
    def __init__(self, ttl: int = 300, maxsize: int = 1024):
        self.ttl = int(ttl)
        self.maxsize = int(maxsize)
        self.store = {}  # key -> (value, timestamp)

    def _prune(self):
        if len(self.store) <= self.maxsize:
            return
        items = sorted(self.store.items(), key=lambda kv: kv[1][1])
        for k, _ in items[: len(self.store) - self.maxsize]:
            del self.store[k]

    def get(self, key: str):
        v = self.store.get(key)
        if not v:
            return None
        val, ts = v
        if time.time() - ts > self.ttl:
            del self.store[key]
            return None
        return val

    def set(self, key: str, value: Any):
        self.store[key] = (value, time.time())
        self._prune()
