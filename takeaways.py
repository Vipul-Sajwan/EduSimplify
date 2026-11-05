"""
EduSimplify - Key Takeaways extraction using IBM watsonx.ai (Granite).

Given simplified content, extracts up to 5 concise bullet-point takeaways
that a student should remember.

Return format (unchanged): list[str]
"""

import copy
import hashlib
import logging
import re

from modules.simplifier import _generate, LRUCache

logger = logging.getLogger("edusimplify.takeaways")

NUM_TAKEAWAYS = 5
TAKEAWAYS_MAX_TOKENS = 800    # 5 sentences; Hindi needs more tokens than English
MAX_SOURCE_CHARS = 6000       # how much of the simplified text is sent to the model
MAX_ATTEMPTS = 2              # one retry if fewer than NUM_TAKEAWAYS come back
# Greedy decoding repeats itself, so a retry must sample to give a different result.
RETRY_OVERRIDES = {"decoding_method": "sample", "temperature": 0.7}

_cache = LRUCache(64)

# "- item", "• item", "* item", "– item", "1. item", "2) item"
_BULLET_RE = re.compile(r"^(?:[-•*–—]|\d{1,2}[.)])\s+")

# A bullet ending like this is a finished sentence (includes the Hindi danda "।")
_SENTENCE_END_RE = re.compile(r"[.!?।:;]['\"”’)\]]*$")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def _truncate_source(text: str, limit: int = MAX_SOURCE_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return cut[:space] if space > limit * 0.9 else cut


def _build_takeaways_prompt(simplified_text: str, language: str) -> str:
    return (
        "You are EduSimplify, an educational study-aid assistant.\n"
        f"Output language: {language}\n\n"
        f"From the educational content below, extract exactly {NUM_TAKEAWAYS} "
        "key takeaways a student must remember.\n\n"
        "Rules:\n"
        "- Each takeaway must be one clear, self-contained sentence on a single line.\n"
        "- Cover the most important concepts; avoid trivial details.\n"
        "- Do not repeat the same idea twice.\n"
        "- Start every takeaway with a dash and a space: '- '\n"
        "- Output ONLY the bullet list, with no introduction and no closing remarks.\n\n"
        f"Educational content:\n{_truncate_source(simplified_text)}\n\n"
        "Key Takeaways:"
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _tidy(item: str) -> str:
    item = re.sub(r"\*\*|__|`", "", item)           # markdown bold / code markers
    return re.sub(r"\s+", " ", item).strip()


def _parse_bullets(raw: str) -> list[str]:
    """
    Extract takeaways from model output.

    - Accepts -, •, *, – and numbered bullets
    - Joins a bullet that the model wrapped onto the next line
    - Ignores intro/closing text ("Here are the key takeaways:")
    - Falls back to plain lines if the model used no bullets at all
    - Removes duplicates
    """
    raw = re.sub(r"```\w*", "", raw or "")

    items, plain = [], []
    current = None                          # index of the bullet being continued

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            current = None                  # blank line ends a bullet
            continue
        if _BULLET_RE.match(line):
            items.append(_BULLET_RE.sub("", line, count=1))
            current = len(items) - 1
        elif current is not None and not _SENTENCE_END_RE.search(items[current]):
            items[current] += " " + line    # wrapped continuation (sentence not finished yet)
        else:
            current = None
            plain.append(line)              # intro / closing text outside any bullet

    if not items:
        # No bullets at all: use plain lines, skipping headers like "Key takeaways:"
        items = [p for p in plain if not p.endswith(":")]

    result, seen = [], set()
    for item in items:
        item = _tidy(item)
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result[:NUM_TAKEAWAYS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_takeaways(simplified_text: str, language: str = "English",
                       use_cache: bool = True) -> list[str]:
    """
    Returns a list of up to NUM_TAKEAWAYS plain-text takeaway strings,
    or [] if none could be generated.
    Set use_cache=False to force a fresh result (e.g. a "Regenerate" button).
    """
    text = (simplified_text or "").strip()
    if not text:
        return []

    cache_key = hashlib.sha256(f"takeaways|{language}|{text}".encode("utf-8")).hexdigest()
    if use_cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            logger.info("Takeaways cache hit.")
            return copy.deepcopy(cached)

    prompt = _build_takeaways_prompt(text, language)
    best = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        overrides = RETRY_OVERRIDES if attempt > 1 else {}
        try:
            raw = _generate(prompt, TAKEAWAYS_MAX_TOKENS, **overrides)
        except Exception:
            if attempt == 1:
                raise                       # nothing generated yet: let the caller see the error
            logger.warning("Takeaways retry failed; keeping the first attempt.", exc_info=True)
            break

        parsed = _parse_bullets(raw)
        if len(parsed) > len(best):
            best = parsed
        if len(best) >= NUM_TAKEAWAYS:
            break

        logger.warning("Takeaways attempt %d gave %d item(s). Raw output (first 500 chars):\n%s",
                       attempt, len(parsed), (raw or "")[:500])

    if best and use_cache:
        _cache.set(cache_key, copy.deepcopy(best))

    return best