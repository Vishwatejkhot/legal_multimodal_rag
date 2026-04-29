from __future__ import annotations
import base64
from pathlib import Path
import anthropic
from config import ANTHROPIC_API_KEY, VISION_MODEL
from reliability.rate_limiter import throttle, anthropic_retry
from reliability.cache import cached
from reliability.logger import log

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_SYSTEM = (
    "You are a legal document analyst. "
    "When given an image of a legal document, photo, letter, or form, extract ALL text verbatim, "
    "then summarise: document type, key dates, named parties, and any legal references. "
    "If handwritten, transcribe as accurately as possible. Flag illegible parts."
)

@cached
@anthropic_retry
def describe_image(image_path: str, question: str = "") -> str:
    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "gif": "image/gif"}.get(suffix, "image/jpeg")

    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()

    prompt = question if question else "Extract all text and summarise this legal document."
    throttle()
    log.info("vision_request", file=str(path), model=VISION_MODEL)

    response = _client.messages.create(
        model=VISION_MODEL,
        max_tokens=2000,
        system=_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text",  "text": prompt},
            ],
        }],
    )
    result = response.content[0].text if response.content else ""
    log.info("vision_response", file=str(path), chars=len(result))
    return result
