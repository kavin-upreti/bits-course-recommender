"""Catch course codes in the LLM's reply that no tool returned."""
import re

from .codes import find_codes
from .history import with_equivalents

SORRY = "Sorry, I couldn't put together a reliable answer. Please try rephrasing your request."
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def allowed_codes(seen: set[str], student_codes: set[str]) -> set[str]:
    """Codes the reply may mention: from tool results or the student's own message, plus their equivalents."""
    return with_equivalents(seen | student_codes)


def disallowed(text: str, allowed: set[str]) -> list[str]:
    """Codes in the text that aren't allowed, in order of appearance."""
    return [code for code in find_codes(text) if code not in allowed]


def strip_codes(text: str, bad: list[str]) -> str:
    """Drop every sentence (or line) that mentions a bad code; SORRY if nothing with words is left."""
    kept_lines = []
    for line in text.splitlines():
        sentences = [sentence for sentence in SENTENCE.split(line) if not set(find_codes(sentence)) & set(bad)]
        if sentences or not line.strip():
            kept_lines.append(" ".join(sentences))
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
    return result if re.search(r"[A-Za-z]{2,}", result) else SORRY
