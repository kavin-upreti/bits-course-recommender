"""A course's text, cut into short clean pieces for embedding.

One embedding per piece (not per handout): a query about "transformers" matches the one topic line about
transformers instead of being diluted by a whole handout.
"""
import re

from catalog.models import Course, Handout
from students.templatetags.text import course_title

from . import config

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
# leading numbering / labels, stripped repeatedly: "1.", "1)", "(a)", "i.", "L1-3", "L(15-16)", "Lec 4-6:",
# "Lecture 7", "Week 2", "Module 3:", "Unit II:", "CLO2.", bare lecture numbers ("7-9 Biotechnology"), bullets
LABEL = re.compile(
    r"^\s*(?:"
    r"[•\-\*·▪◦●–]+\s*"
    r"|\(?\d+(?:\.\d+)*[.)](?!\d)\s*"
    r"|\([a-z]\)\s*|[a-z][.)]\s+"
    r"|\(?(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)[.)]\s+"
    r"|L\d+(?:\s*[-–]\s*L?\d+)\s*[:.\-–]?\s*|L\d+\s*[:.\-–]\s*|L\(\d+(?:\s*[-–]\s*\d+)?\)\s*"
    r"|C?L?O\d+\s*[.:)\-–]\s*"
    r"|\d+(?:\s*[-–]\s*\d+)?\s+(?=(?-i:[A-Z]))"
    r"|(?:lec(?:ture)?s?\.?|week|module|unit|session)\s*[\dIVX]+(?:\s*[-–]\s*[\dIVX]+)?\s*[:.\-–]?\s+"
    r")",
    re.I,
)
# bare section headers are format words, not content
HEADERS = {"topics", "course plan", "learning objectives", "learning outcomes", "course description",
           "scope and objective", "scope and objectives", "objectives"}

Piece = tuple[str, str, str]  # (kind, text, source)


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends followed by a capital / digit / bracket."""
    return [sentence.strip() for sentence in SENTENCE_END.split(" ".join(text.split())) if sentence.strip()]


def clean(text: str) -> str:
    """Strip leading labels and collapse whitespace."""
    text = " ".join(text.split())
    while True:
        stripped = LABEL.sub("", text, count=1).strip()
        if stripped == text:
            return text
        text = stripped


def _key(text: str) -> str:
    """Dedupe key: case and trailing punctuation don't make a piece different."""
    return text.lower().rstrip(" .;:")


def _keep(text: str) -> bool:
    return len(text.split()) >= config.PIECE_MIN_WORDS and text.lower().rstrip(":.") not in HEADERS


def _cut(text: str) -> list[str]:
    """A cleaned piece, or its sentences if it's too long (a sentence still too long is kept whole)."""
    text = clean(text)
    parts = split_sentences(text) if len(text.split()) > config.PIECE_MAX_WORDS else [text]
    return [clean(part) for part in parts]


def _handout_pieces(handout: Handout) -> list[Piece]:
    raw = [("description", sentence) for sentence in split_sentences(handout.description)]
    raw += [("topic", topic) for topic in handout.topics]
    raw += [("outcome", outcome) for outcome in handout.learning_outcomes]
    return [(kind, part, handout.file) for kind, text in raw for part in _cut(text)]


def build_pieces(course: Course) -> list[Piece]:
    """Title, then each handout's description sentences / topics / outcomes (handouts by file name), then the
    Bulletin description's sentences. Cleaned, short fragments and headers dropped, deduplicated case-insensitively."""
    pieces: list[Piece] = []
    seen: set[str] = set()
    title = course_title(course.title).strip()
    if title:  # why no length rule: the title is always kept, even one word ("Thermodynamics")
        pieces.append(("title", title, "timetable"))
        seen.add(_key(title))
    candidates = [piece for handout in sorted(course.handouts.all(), key=lambda handout: handout.file)
                  for piece in _handout_pieces(handout)]
    candidates += [("bulletin_description", part, "bulletin")
                   for sentence in split_sentences(course.description) for part in _cut(sentence)]
    for kind, text, source in candidates:
        if _keep(text) and _key(text) not in seen:
            seen.add(_key(text))
            pieces.append((kind, text, source))
    return pieces
