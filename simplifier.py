"""
EduSimplify - content simplification using IBM watsonx.ai

  1. Model client is created once (thread-safe) and reused
  2. Bounded, thread-safe LRU cache for repeated requests
  3. Long text is chunked (no silent truncation) and processed in parallel
  4. Retries with exponential backoff
  5. Lean prompt
  6. Dynamic max_new_tokens per chunk
  7. Optional streaming for UIs (Streamlit / Flask / FastAPI)
  8. Latency logging
"""

import os
import re
import time
import hashlib
import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edusimplify")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
URL = os.getenv("WATSONX_URL")
MODEL_ID = os.getenv("WATSONX_MODEL_ID")

MAX_INPUT_CHARS = 60_000      # hard safety cap on total input
CHUNK_CHARS = 4_000           # target size of each chunk
MAX_WORKERS = 4               # keep modest to avoid rate limits
CACHE_SIZE = 256              # number of results kept in memory
MIN_NEW_TOKENS = 200
MAX_NEW_TOKENS = 1000

# Passed on EVERY call, because per-call params replace (not merge with) the
# ones given at init. Note: temperature only has an effect when
# decoding_method is "sample"; the default is greedy (deterministic).
BASE_PARAMS = {
    "temperature": 0.3,
    "repetition_penalty": 1.05,
}


def _params(max_new_tokens: int, **overrides) -> dict:
    return {**BASE_PARAMS, "max_new_tokens": max_new_tokens, **overrides}


# ---------------------------------------------------------------------------
# Model (created once, thread-safe, reused)
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()


def get_model() -> ModelInference:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:      # double-checked: only one thread creates it
                if not all([API_KEY, PROJECT_ID, URL, MODEL_ID]):
                    raise RuntimeError("IBM credentials are missing. Check your .env file.")
                logger.info("Initializing watsonx model: %s", MODEL_ID)
                _model = ModelInference(
                    model_id=MODEL_ID,
                    credentials={"url": URL, "apikey": API_KEY},
                    project_id=PROJECT_ID,
                    params=_params(MAX_NEW_TOKENS),
                )
    return _model


# ---------------------------------------------------------------------------
# Thread-safe LRU cache
# ---------------------------------------------------------------------------
class LRUCache:
    def __init__(self, max_size: int):
        self._data = OrderedDict()
        self._max = max_size
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def set(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)


_cache = LRUCache(CACHE_SIZE)


def _cache_key(content: str, level: str, language: str) -> str:
    raw = f"{MODEL_ID}|{level}|{language}|{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------
def _split_long_paragraph(paragraph: str, max_chars: int):
    """Split an oversized paragraph on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    parts, current = [], ""
    for s in sentences:
        # A single sentence longer than max_chars is hard-split
        while len(s) > max_chars:
            if current:
                parts.append(current.strip())
                current = ""
            parts.append(s[:max_chars])
            s = s[max_chars:]
        if len(current) + len(s) + 1 > max_chars and current:
            parts.append(current.strip())
            current = ""
        current += s + " "
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_text(text: str, max_chars: int = CHUNK_CHARS):
    """Split text into chunks on paragraph boundaries."""
    chunks, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue

        pieces = _split_long_paragraph(para, max_chars) if len(para) > max_chars else [para]

        for piece in pieces:
            if len(current) + len(piece) + 2 > max_chars and current:
                chunks.append(current.strip())
                current = ""
            current += piece + "\n\n"

    if current.strip():
        chunks.append(current.strip())
    return chunks


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def build_prompt(content: str, level: str, language: str,
                 part: int = 1, total_parts: int = 1) -> str:
    part_note = ""
    if total_parts > 1:
        part_note = (
            f"\nThis is part {part} of {total_parts} of a longer text. "
            "Do not add an introduction or conclusion.\n"
        )

    return (
        "You are EduSimplify, an educational content simplification agent.\n"
        f"Difficulty level: {level}\n"
        f"Output language: {language}\n"
        f"{part_note}\n"
        "Rules:\n"
        "- Use short, simple sentences.\n"
        "- Define technical terms in plain words.\n"
        "- Add one relevant example for each key concept.\n"
        "- Keep all important academic facts accurate. Do not invent information.\n\n"
        f"Academic content:\n{content}\n\n"
        "Simplified explanation:"
    )


def _token_budget(chunk: str) -> int:
    """Rough output budget: ~1 token per 4 chars of input, plus room for examples."""
    estimate = int(len(chunk) / 4 * 1.2)
    return max(MIN_NEW_TOKENS, min(MAX_NEW_TOKENS, estimate))


# ---------------------------------------------------------------------------
# Generation with retries
# ---------------------------------------------------------------------------
MAX_RETRIES = 3


def _generate(prompt: str, max_new_tokens: int, **overrides) -> str:
    """
    Call the model, retrying with exponential backoff (1s, 2s, 4s...).
    Extra keyword arguments override BASE_PARAMS for this call
    (e.g. decoding_method="sample").
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = get_model().generate_text(
                prompt=prompt,
                params=_params(max_new_tokens, **overrides),
            )
            return str(result)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = min(2 ** (attempt - 1), 8)
            logger.warning("Attempt %d failed (%s). Retrying in %ds...",
                           attempt, exc, wait)
            time.sleep(wait)
    raise RuntimeError("_generate: exhausted retries without returning")  # unreachable


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def simplify_content(content: str, level: str, language: str) -> str:
    """Simplify academic content. Returns the full simplified text."""
    content = (content or "").strip()
    if not content:
        raise ValueError("No content provided.")

    if len(content) > MAX_INPUT_CHARS:
        logger.warning("Input truncated from %d to %d characters.",
                       len(content), MAX_INPUT_CHARS)
        content = content[:MAX_INPUT_CHARS]

    key = _cache_key(content, level, language)
    cached = _cache.get(key)
    if cached is not None:
        logger.info("Cache hit.")
        return cached

    start = time.perf_counter()

    chunks = chunk_text(content)
    total = len(chunks)
    jobs = [
        (build_prompt(c, level, language, i + 1, total), _token_budget(c))
        for i, c in enumerate(chunks)
    ]

    if total == 1:
        results = [_generate(*jobs[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total)) as pool:
            results = list(pool.map(lambda job: _generate(*job), jobs))

    output = "\n\n".join(r.strip() for r in results if r and r.strip())

    _cache.set(key, output)
    logger.info("Simplified %d chars in %d chunk(s) in %.2fs.",
                len(content), total, time.perf_counter() - start)
    return output


def simplify_content_stream(content: str, level: str, language: str):
    """
    Generator that yields text pieces as they are produced.
    Chunks are processed sequentially so output stays in order.

    Streamlit:  st.write_stream(simplify_content_stream(text, level, lang))
    Flask:      return Response(simplify_content_stream(...), mimetype="text/plain")
    """
    content = (content or "").strip()
    if not content:
        raise ValueError("No content provided.")
    content = content[:MAX_INPUT_CHARS]

    key = _cache_key(content, level, language)
    cached = _cache.get(key)
    if cached is not None:
        yield cached
        return

    chunks = chunk_text(content)
    total = len(chunks)
    collected = []

    for i, chunk in enumerate(chunks):
        prompt = build_prompt(chunk, level, language, i + 1, total)
        params = _params(_token_budget(chunk))
        for piece in get_model().generate_text_stream(prompt=prompt, params=params):
            collected.append(piece)
            yield piece
        if i < total - 1:
            collected.append("\n\n")
            yield "\n\n"

    _cache.set(key, "".join(collected).strip())


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Photosynthesis is the biochemical process by which autotrophic organisms "
        "convert light energy into chemical energy stored in glucose, using carbon "
        "dioxide and water, and releasing oxygen as a byproduct."
    )
    print(simplify_content(sample, level="Beginner", language="English"))