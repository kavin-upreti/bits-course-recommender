"""Course cards shown under the reply (todo.md 10.3). Built from the database, never from the LLM's text."""
from catalog.models import Course
from students.templatetags.text import course_title

from . import config
from .codes import find_codes
from .context import StudentContext
from .handout_facts import CourseFacts, attendance_text, get_course_facts, makeup_text, number
from .tools import DISPLAY_CATEGORY, summary


def recommended_codes(reply: str, plans: list[dict], listed: dict[str, dict]) -> list[str]:
    """The sections of the last successful check_plan; else codes in the reply that some `courses` list returned."""
    if plans:
        codes = list(plans[-1]["sections"])
    else:
        codes = [code for code in find_codes(reply) if code in listed]
    return codes[:config.MAX_CARDS]


def _unknown(facts: CourseFacts, name: str) -> str:
    text = facts.unknown_text(name)
    return text[0].upper() + text[1:]


def _yes_no(value: bool | None) -> str:
    return "Yes" if value else "No"


def quiz_text(facts: CourseFacts) -> str | None:
    """"3 (15%)", "Yes (10%)" when the count isn't stated, "None", or None when unknown."""
    percent = f" ({number(facts.quiz_percent)}%)" if facts.quiz_percent else ""
    if facts.quiz_count:
        return f"{facts.quiz_count}{percent}"
    if facts.quiz_percent:
        return f"Yes{percent}"
    return "None" if facts.quiz_count == 0 or facts.quiz_percent == 0 else None


def card_facts(facts: CourseFacts) -> list[dict]:
    """Label / value pairs; unknown values read "Couldn't verify (reason)"."""
    def show(name: str, text) -> str:
        return _unknown(facts, name) if getattr(facts, name) is None else text

    midsem = f"Yes ({number(facts.midsem_percent)}%)" if facts.has_midsem and facts.midsem_percent else _yes_no(facts.has_midsem)
    attendance, makeup, quizzes = attendance_text(facts), makeup_text(facts), quiz_text(facts)
    return [
        {"label": "Midsem", "value": show("has_midsem", midsem)},
        {"label": "Compre", "value": show("compre_percent", f"{number(facts.compre_percent or 0)}%")},
        {"label": "Quizzes", "value": quizzes or _unknown(facts, "quiz_count")},
        {"label": "Project", "value": show("project_percent", f"{number(facts.project_percent or 0)}%" if facts.project_percent else "None")},
        {"label": "Open book", "value": show("open_book", _yes_no(facts.open_book))},
        {"label": "Attendance", "value": attendance.capitalize() if attendance else _unknown(facts, "attendance_required")},
        {"label": "Makeup", "value": makeup[0].upper() + makeup[1:] if makeup else _unknown(facts, "makeup_allowed")},
    ]


def sources(course: Course) -> list[str]:
    """Where the card's facts come from: each handout (with its evaluation page) and the Bulletin."""
    found = []
    for handout in course.handouts.order_by("file"):
        page = ((handout.sources or {}).get("pages") or {}).get("evaluation")
        found.append(f"handout: {handout.file}" + (f", p. {page}" if page else ""))
    if course.description:
        found.append("Bulletin")
    return found


def ltpu(course: Course) -> str:
    return "-".join("?" if value is None else str(value) for value in (course.L, course.T, course.P, course.units))


def build_card(ctx: StudentContext, course: Course, listed: dict | None, sections: dict | None) -> dict:
    """One card; category / counts_as / note from the latest tool result that had the course."""
    category = ctx.category_map.get(course.code, "")
    card = {
        "code": course.code, "title": course_title(course.title),
        "category": (listed or {}).get("category") or DISPLAY_CATEGORY.get(category, category),
        "counts_as": (listed or {}).get("counts_as"), "note": (listed or {}).get("note"),
        "units": course.units, "ltpu": ltpu(course),
        "facts": card_facts(get_course_facts(course)), "description": summary(course), "sources": sources(course),
        "score": (listed or {}).get("score"),  # match / penalty / final / why, when a search returned the course
        "also_offered_as": (listed or {}).get("also_offered_as") or [],
    }
    if sections:
        card["sections"] = sections
    return card


def build_cards(ctx: StudentContext, reply: str, plans: list[dict], listed: dict[str, dict],
                mentioned: dict[str, dict]) -> list[dict]:
    """listed: code -> latest `courses` entry; mentioned: code -> latest entry in any result list (for the labels)."""
    codes = recommended_codes(reply, plans, listed)
    courses = {course.code: course for course in Course.objects.filter(code__in=codes)}
    plan_sections = plans[-1]["sections"] if plans else {}
    return [build_card(ctx, courses[code], mentioned.get(code), plan_sections.get(code) or mentioned.get(code, {}).get("sections"))
            for code in codes if code in courses]
