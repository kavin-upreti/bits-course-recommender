"""Everything the tools need about the logged-in student, built once per chat message (todo.md section 5.1).

Nothing in here is ever sent to the LLM: the tools read it and return only small results.
"""
from dataclasses import dataclass, field

from catalog.models import KnownGap, MinorCourse, Offering
from students.models import Student, StudentCourse

from .categories import ELECTIVE_CATEGORIES, category_map
from .history import done_codes
from .requirements import elective_needs, semester_load


@dataclass
class CategoryStatus:
    """One elective requirement (a dual degree's two DEL counts added up)."""

    required_courses: int
    required_units: int
    done_courses: int
    done_units: int

    @property
    def remaining_courses(self) -> int:
        return max(0, self.required_courses - self.done_courses)

    @property
    def remaining_units(self) -> int:
        return max(0, self.required_units - self.done_units)

    @property
    def complete(self) -> bool:
        return self.remaining_courses == 0 and self.remaining_units == 0


@dataclass
class StudentContext:
    student: Student
    semester_tag: str                    # the timetable's semester (from Offering), never hardcoded
    completed: set[str]                  # completed codes + all their equivalents / mappings
    current: set[str]                    # current codes + all their equivalents / mappings
    current_courses: list[str]           # the student's current course codes as entered (for check_plan)
    category_map: dict[str, str]         # code -> AUDIT / GIR / CDC / CHART / DEL / HUEL / OPEL
    remaining: dict[str, CategoryStatus] # HUEL / DEL / OPEL; empty when known_gap is set
    known_gap: str | None                # why requirements can't be computed (2026+ batch), else None
    default_avoid_8am: bool
    default_avoid_day: str | None
    avoid_eval_styles: list[str]
    interests: list[str]
    minor_courses: dict[str, str] = field(default_factory=dict)  # code -> "core" | "elective"
    max_units: int | None = None         # from the Rule table; None if not stated for this batch
    timings: list[dict] = field(default_factory=list)  # per get_eligible_courses call, for the debug panel

    @property
    def is_dual(self) -> bool:
        return self.student.programme.type == "dual"


def resolve_preferences(ctx: StudentContext, avoid_8am: bool | None, avoid_day: str | None) -> dict:
    """Message values win over profile defaults; remember where each came from."""
    return {
        "avoid_8am": ctx.default_avoid_8am if avoid_8am is None else avoid_8am,
        "avoid_8am_source": "profile" if avoid_8am is None else "message",
        "avoid_day": ctx.default_avoid_day if avoid_day is None else avoid_day,
        "avoid_day_source": "profile" if avoid_day is None else "message",
    }


def batch_gap(student: Student) -> str | None:
    """The user message of a KnownGap covering the student's batch ("2026 onwards"): requirements can't be computed.
    ponytail: same "<year> onwards" parsing as students.views.applicable_gaps; other batch shapes aren't in the data."""
    for gap in KnownGap.objects.exclude(user_message__isnull=True).exclude(user_message="").order_by("gap_id"):
        batches = " ".join(gap.affected.get("batches", []))
        if batches.endswith("onwards") and student.admission_year >= int(batches.split()[0]):
            return gap.user_message
    return None


def requirement_status(student: Student) -> dict[str, CategoryStatus]:
    """HUEL / DEL / OPEL from elective_needs. Dual degrees: both DEL counts added; no OPEL requirement (the degree
    audit has none), so OPEL is 0 of 0, i.e. complete."""
    status = {category: CategoryStatus(0, 0, 0, 0) for category in ELECTIVE_CATEGORIES}
    for need in elective_needs(student):
        total = status[need.category]
        total.required_courses += need.required_courses
        total.required_units += need.required_units
        total.done_courses += min(need.done_courses, need.required_courses)
        total.done_units += min(need.done_units, need.required_units)
    return status


def build_context(student: Student) -> StudentContext:
    """Fill the context from the existing requirement / history / category helpers."""
    known_gap = batch_gap(student)
    minor = {}
    if student.minor_id:
        minor = dict(MinorCourse.objects.filter(minor_id=student.minor_id).values_list("course__code", "role"))
    return StudentContext(
        student=student,
        semester_tag=Offering.objects.order_by("pk").values_list("semester_tag", flat=True).first() or "",
        completed=done_codes(student, "completed"),
        current=done_codes(student, "current"),
        current_courses=sorted(StudentCourse.objects.filter(student=student, status="current")
                               .values_list("course__code", flat=True)),
        category_map=category_map(student.programme),
        remaining={} if known_gap else requirement_status(student),
        known_gap=known_gap,
        default_avoid_8am=student.default_avoid_8am,
        default_avoid_day=student.default_avoid_day or None,
        avoid_eval_styles=list(student.avoid_eval_styles),
        interests=list(student.interests),
        minor_courses=minor,
        max_units=semester_load(student).max_units,
    )
