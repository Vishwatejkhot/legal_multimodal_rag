import hashlib
import json
import diskcache
from config import CACHE_DIR, CACHE_TTL_SECONDS

_cache = diskcache.Cache(str(CACHE_DIR))

def make_key(*args, **kwargs) -> str:
    payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()

def get(key: str):
    return _cache.get(key)

def set(key: str, value, ttl: int = CACHE_TTL_SECONDS):
    _cache.set(key, value, expire=ttl)

def cached(fn):
    """Decorator that caches the return value of a function for 24 hours."""
    def wrapper(*args, **kwargs):
        key = f"{fn.__module__}.{fn.__qualname__}:{make_key(*args, **kwargs)}"
        hit = get(key)
        if hit is not None:
            return hit
        result = fn(*args, **kwargs)
        set(key, result)
        return result
    return wrapper
