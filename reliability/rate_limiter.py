import time
import threading
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import anthropic

class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens: float = 1.0):
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
            self._last = now
            if self._tokens < tokens:
                time.sleep((tokens - self._tokens) / self._rate)
                self._tokens = 0
            else:
                self._tokens -= tokens

_bucket = TokenBucket(rate=50 / 60, capacity=50)

def throttle():
    _bucket.consume()

anthropic_retry = retry(
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)

# Keep alias so existing imports of openai_retry still work
openai_retry = anthropic_retry
