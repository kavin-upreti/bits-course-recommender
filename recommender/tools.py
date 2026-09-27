"""The four tools the LLM can call (todo.md sections 5-7). Each takes the StudentContext plus the LLM's (already
validated) arguments and returns a small JSON-serialisable dict. Academic rules live here and in the modules these
call, never in the LLM."""
from catalog.models import Course
from students.templatetags.text import course_title

from . import config
from .codes import normalise_code, resolve_course
from .context import StudentContext
from .eligible import run
from .handout_facts import CourseFacts, attendance_text, get_course_facts, makeup_text, number
from .pieces import split_sentences
from .plan import check_plan, offered  # noqa: F401  (check_plan is one of the four tools)

DISPLAY_CATEGORY = {"CHART": "CDC"}  # compulsory chart courses read as CDCs (same as the UI)
DUAL_DEL_NOTE = "dual degree: DEL counts are for both degrees combined"


# ---------------------------------------------------------------- 5.2

def get_remaining_requirements(ctx: StudentContext) -> dict:
    """Remaining HUEL / DEL / OPEL courses and units."""
    if ctx.known_gap:
        return {"error": ctx.known_gap}
    result: dict = {category: {"remaining_courses": status.remaining_courses, "remaining_units": status.remaining_units,
                               "complete": status.complete}
                    for category, status in ((name, ctx.remaining[name]) for name in ("HUEL", "DEL", "OPEL"))}
    if ctx.is_dual:
        result["note"] = DUAL_DEL_NOTE
    return result


# ---------------------------------------------------------------- 5.3

def summary(course: Course) -> str:
    """First sentences of the first handout description, else of the Bulletin description."""
    texts = [handout.description for handout in course.handouts.order_by("file") if handout.description.strip()]
    sentences = split_sentences(texts[0] if texts else course.description)
    return " ".join(sentences[:config.SUMMARY_SENTENCES])


def yes_no(value: bool | None, facts: CourseFacts, name: str) -> str:
    return facts.unknown_text(name) if value is None else ("yes" if value else "no")


def detail_facts(facts: CourseFacts) -> dict:
    """The handout facts in words; unknown ones say "couldn't verify"."""
    known = lambda name: facts.unknown_text(name) if getattr(facts, name) is None else number(getattr(facts, name))  # noqa: E731
    return {
        "midsem": yes_no(facts.has_midsem, facts, "has_midsem"),
        "quizzes": known("quiz_count"),
        "compre_percent": known("compre_percent"),
        "project_percent": known("project_percent"),
        "open_book": yes_no(facts.open_book, facts, "open_book"),
        "attendance": attendance_text(facts) or facts.unknown_text("attendance_required"),
        "makeup": makeup_text(facts) or facts.unknown_text("makeup_allowed"),
    }


def prerequisites_met(course: Course, ctx: StudentContext) -> bool | str:
    if course.needs_verification and not course.prerequisites:
        return "couldn't verify"
    return all(set(group) & ctx.completed for group in course.prerequisites or [])


def get_course_details(ctx: StudentContext, code: str) -> dict:
    """Compact facts about one course. The only tool that returns description text (two sentences)."""
    course = resolve_course(code)
    if course is None:
        return {"error": f"No course with code {normalise_code(code)}"}
    category = ctx.category_map.get(course.code, "")
    return {
        "code": course.code, "title": course_title(course.title),
        "category": DISPLAY_CATEGORY.get(category, category), "units": course.units,
        "prerequisites": [" or ".join(group) for group in course.prerequisites or []],
        "prerequisites_met": prerequisites_met(course, ctx),
        "offered_this_semester": offered(course, ctx.semester_tag),
        "facts": detail_facts(get_course_facts(course)),
        "summary": summary(course),
    }


# ---------------------------------------------------------------- 6

def get_eligible_courses(ctx: StudentContext, category: str | None = None, about: list[str] | str | None = None,
                         filters: dict | None = None, avoid_8am: bool | None = None, avoid_day: str | None = None,
                         exclude: list[str] | None = None, count: int | None = None,
                         related: list[str] | str | None = None) -> dict:
    """Eligible courses, best match first (see recommender/eligible.py)."""
    return run(ctx, category, about, filters, avoid_8am, avoid_day, exclude, count, related)[0]
