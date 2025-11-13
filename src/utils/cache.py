import hashlib
import json
import os
from typing import Any, Dict, Optional


def _stable_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ContentHashCache:
    def __init__(self, cache_dir: str = ".cache") -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.db_path = os.path.join(self.cache_dir, "responses.jsonl")
        # Index in-memory for quick lookup
        self._index: Dict[str, int] = {}
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        record = json.loads(line)
                        self._index[record["key"]] = i
                    except Exception:
                        continue

    def key_for(self, payload: Dict[str, Any]) -> str:
        return _stable_hash(payload)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key not in self._index:
            return None
        target_line = self._index[key]
        with open(self.db_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == target_line:
                    try:
                        return json.loads(line)
                    except Exception:
                        return None
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        record = {"key": key, **value}
        with open(self.db_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Update index to last line
        self._index[key] = self._index.get(key, len(self._index))

