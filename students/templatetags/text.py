"""Display filters for catalog text: course titles and Bulletin descriptions."""
import re

from django import template

register = template.Library()

SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "via", "with"}
ROMAN = re.compile(r"^(i{1,3}|iv|vi{0,3}|ix|x)$", re.I)  # I..X: course sequence numbers ("Mathematics III")


def _word(word: str, first: bool) -> str:
    core = word.strip("()[],:;.&")
    if ROMAN.match(core):
        return word.upper()
    if not first and word.lower() in SMALL_WORDS:
        return word.lower()
    # capitalise each hyphen part: "WELL-BEING" -> "Well-Being"
    return "-".join(part[:1].upper() + part[1:].lower() for part in word.split("-"))


@register.filter
def course_title(text: str) -> str:
    """ALL-CAPS timetable titles in title case, keeping Roman numerals ("MATHEMATICS III" -> "Mathematics III").
    Titles that already have their own casing (so acronyms like "IoT" survive) are returned unchanged."""
    if not text or text != text.upper():
        return text
    return " ".join(_word(word, i == 0) for i, word in enumerate(text.split()))


@register.filter
def as_points(text: str) -> list[str]:
    """A Bulletin description ("topic; topic; topic.") as bullet points, or [] if it reads better as a paragraph."""
    parts = [part.strip(" .") for part in re.split(r";\s*", text or "") if part.strip(" .")]
    return [part[0].upper() + part[1:] for part in parts] if len(parts) >= 3 else []
