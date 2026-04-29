import hashlib
import json
from pathlib import Path

_SEEN_FILE = Path(__file__).parent.parent / "indexes" / "seen_hashes.json"

def _load() -> set:
    if _SEEN_FILE.exists():
        return set(json.loads(_SEEN_FILE.read_text()))
    return set()

def _save(seen: set):
    _SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SEEN_FILE.write_text(json.dumps(list(seen)))

def is_duplicate(text: str) -> bool:
    seen = _load()
    h = hashlib.md5(text.encode()).hexdigest()
    if h in seen:
        return True
    seen.add(h)
    _save(seen)
    return False
