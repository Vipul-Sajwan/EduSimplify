"""
EduSimplify - Quiz / MCQ generation using IBM watsonx.ai (Granite).

Given simplified content, generates up to 5 multiple-choice questions with
4 options each and the correct answer marked.

Return format (unchanged):
    [{"question": str, "options": {"A": str, "B": str, "C": str, "D": str}, "answer": "A"}, ...]
"""

import copy
import hashlib
import json
import logging
import random
import re

from modules.simplifier import _generate, LRUCache

logger = logging.getLogger("edusimplify.quiz")

NUM_QUESTIONS = 5
QUIZ_MAX_TOKENS = 1500        # 5 MCQs as JSON need more room than a summary (Hindi needs even more)
MAX_SOURCE_CHARS = 6000       # how much of the simplified text is sent to the model
MAX_ATTEMPTS = 2              # one retry if too few valid questions come back
# Greedy decoding gives the same output every time, so a retry must sample to differ.
RETRY_OVERRIDES = {"decoding_method": "sample", "temperature": 0.7}
LETTERS = ("A", "B", "C", "D")

_cache = LRUCache(64)
_decoder = json.JSONDecoder()


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def _truncate_source(text: str, limit: int = MAX_SOURCE_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return cut[:space] if space > limit * 0.9 else cut


def _build_quiz_prompt(simplified_text: str, language: str) -> str:
    return (
        "You are EduSimplify, an educational quiz generator.\n"
        f"Output language: {language}\n\n"
        f"Based on the educational content below, generate exactly {NUM_QUESTIONS} "
        "multiple-choice questions.\n\n"
        "Rules:\n"
        "- Each question must have exactly 4 options labeled A, B, C, D.\n"
        "- Exactly one option is correct. Set \"answer\" to its letter.\n"
        "- Questions must test understanding, not just word recall.\n"
        "- Do not use options like \"All of the above\", \"None of the above\" "
        "or \"Both A and B\".\n"
        "- Write the question and option texts in the output language, but keep the "
        "JSON keys and the answer letter in English.\n"
        "- Return ONLY valid JSON, with no markdown fences and no extra text.\n\n"
        "Required JSON format:\n"
        "[\n"
        "  {\n"
        '    "question": "...",\n'
        '    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},\n'
        '    "answer": "A"\n'
        "  }\n"
        "]\n\n"
        f"Educational content:\n{_truncate_source(simplified_text)}\n\n"
        "JSON:"
    )


# ---------------------------------------------------------------------------
# JSON extraction (robust to prose, code fences and truncated output)
# ---------------------------------------------------------------------------
def _extract_questions(raw: str) -> list:
    """
    Pull every complete question object out of the model output.

    Unlike parsing one big [...] block, this still recovers the finished
    questions when the output is cut off by the token limit mid-way.
    """
    raw = re.sub(r"```(?:json)?", "", raw or "")
    found = []
    i = 0
    while True:
        i = raw.find("{", i)
        if i == -1:
            break
        try:
            obj, end = _decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and "question" in obj:
            found.append(obj)
            i = end                 # skip past it so nested "options" isn't re-scanned
        else:
            i += 1                  # e.g. a wrapper object; keep scanning inside it
    return found


# ---------------------------------------------------------------------------
# Validation / normalization
# ---------------------------------------------------------------------------
def _letter(text: str):
    match = re.search(r"\b([A-D])\b", str(text).upper())
    return match.group(1) if match else None


def _normalize(q: dict):
    """Return a clean question dict, or None if it can't be repaired."""
    question = str(q.get("question", "")).strip()
    if not question:
        return None

    options = q.get("options")
    if isinstance(options, list):
        options = dict(zip(LETTERS, options))
    if not isinstance(options, dict):
        return None

    clean = {}
    for key, value in options.items():
        letter = _letter(key)
        text = re.sub(r"^\(?[A-Da-d][\).:]\s+", "", str(value).strip())
        if letter and text:
            clean[letter] = text
    if set(clean) != set(LETTERS):
        return None
    if len({v.casefold() for v in clean.values()}) != 4:   # duplicate options
        return None

    answer = str(q.get("answer", "")).strip()
    correct = next((k for k, v in clean.items() if v.casefold() == answer.casefold()), None)
    if correct is None and len(answer) <= 10:
        correct = _letter(answer)
    if correct not in clean:
        return None

    return {
        "question": question,
        "options": {k: clean[k] for k in LETTERS},
        "answer": correct,
    }


def _shuffle_options(q: dict) -> dict:
    """LLMs favor 'A' and 'B' as the correct answer; randomize the position."""
    correct_text = q["options"][q["answer"]]
    texts = list(q["options"].values())
    random.shuffle(texts)
    options = dict(zip(LETTERS, texts))
    answer = next(k for k, v in options.items() if v == correct_text)
    return {"question": q["question"], "options": options, "answer": answer}


def _validate(candidates: list) -> list:
    valid, seen = [], set()
    for q in candidates:
        if not isinstance(q, dict):
            continue
        fixed = _normalize(q)
        if fixed is None:
            continue
        key = fixed["question"].casefold()
        if key in seen:                                    # duplicate question
            continue
        seen.add(key)
        valid.append(fixed)
    return valid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_quiz(simplified_text: str, language: str = "English",
                  shuffle: bool = True, use_cache: bool = True) -> list[dict]:
    """
    Returns a list of dicts, each with keys:
      question (str), options (dict A-D), answer (str)

    Returns [] if no valid questions could be generated.
    Set use_cache=False to force a fresh set of questions (e.g. a "Regenerate" button).
    """
    text = (simplified_text or "").strip()
    if not text:
        return []

    cache_key = hashlib.sha256(f"quiz|{language}|{shuffle}|{text}".encode("utf-8")).hexdigest()
    if use_cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            logger.info("Quiz cache hit.")
            return copy.deepcopy(cached)

    prompt = _build_quiz_prompt(text, language)
    best = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        overrides = RETRY_OVERRIDES if attempt > 1 else {}
        try:
            raw = _generate(prompt, QUIZ_MAX_TOKENS, **overrides)
        except Exception:
            if attempt == 1:
                raise                      # nothing generated yet: let the caller see the error
            logger.warning("Quiz retry failed; keeping the first attempt.", exc_info=True)
            break
        valid = _validate(_extract_questions(raw))

        if len(valid) > len(best):
            best = valid
        if len(best) >= NUM_QUESTIONS:
            break

        logger.warning("Quiz attempt %d gave %d valid question(s). Raw output (first 500 chars):\n%s",
                       attempt, len(valid), (raw or "")[:500])

    questions = best[:NUM_QUESTIONS]
    if shuffle:
        questions = [_shuffle_options(q) for q in questions]

    if questions and use_cache:
        _cache.set(cache_key, copy.deepcopy(questions))

    return questions