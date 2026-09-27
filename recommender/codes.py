"""Course-code normalisation and lookup, shared by the tools, the guardrail and the cards."""
import re

from catalog.models import CodeMapping, Course, CourseEquivalent

# "cs f425", "CS  F425", "CSF425" -> "CS F425"
CODE = re.compile(r"\b([A-Z]{2,5})\s?([A-Z]\d{3}[A-Z]?)\b")


def normalise_code(raw: str) -> str:
    """Uppercase, one space between department and number; anything that isn't code-shaped is only uppercased."""
    text = " ".join(str(raw).upper().split())
    match = CODE.fullmatch(text)
    return f"{match[1]} {match[2]}" if match else text


def find_codes(text: str) -> list[str]:
    """Every code-shaped token in a text, normalised, in order of first appearance."""
    return list(dict.fromkeys(f"{dept} {number}" for dept, number in CODE.findall(text)))


def resolve_course(code: str) -> Course | None:
    """The Course for a code: itself, or the course a code mapping / timetable equivalent points to."""
    code = normalise_code(code)
    course = Course.objects.filter(code=code).first()
    if course:
        return course
    mapping = CodeMapping.objects.filter(from_code=code).select_related("to_course").first()
    if mapping:
        return mapping.to_course
    equivalent = CourseEquivalent.objects.filter(equivalent_code=code).select_related("course").first()
    return equivalent.course if equivalent else None
