"""Infer a student's compulsory courses from their programme chart (ideation 6.2) and store them as StudentCourse rows.

Everything named in the chart before the planned semester = completed; named in the planned semester = current.
Elective slots are skipped (the student types those in). PS-I / PS-II / thesis are skipped for now.
"""
from catalog.models import CodeMapping, Course, CourseEquivalent, PatternSlot, Programme
from students.models import Student, StudentCourse

from .categories import category_map, chart_slots


def practice_school_codes(programme: Programme) -> set[str]:
    """PS-I, PS-II and thesis codes, read from the programme's own summer / final-year data (not hardcoded).
    A dual degree has two sets (one per degree)."""
    parts = [programme, programme.first_component, programme.second_component]
    codes: set[str] = set()
    for part in filter(None, parts):
        codes.update((part.summer or {}).get("codes", []))
        for option in part.final_year_options or []:
            codes.add(option.get("code"))
            codes.update(option.get("alternatives", []))
    return codes


def course_slots(programme: Programme) -> list[PatternSlot]:
    """Chart slots that are real courses: named, resolved to a Course, and not PS / thesis."""
    skip = practice_school_codes(programme)
    return [slot for slot in chart_slots(programme) if slot.course and slot.course.code not in skip]


def alternative_slots(programme: Programme) -> dict[str, list[dict]]:
    """"X or Y" slots per semester: {"2-2": [{"key": "ECON F211", "options": [Course, Course]}]}.
    The key is the code the chart lists first; it's what Student.alternative_choices is keyed by."""
    codes = {code for slot in course_slots(programme) for code in slot.alternatives}
    courses = {course.code: course for course in Course.objects.filter(code__in=codes)}
    by_semester: dict[str, list[dict]] = {}
    for slot in course_slots(programme):
        options = [slot.course] + [courses[code] for code in slot.alternatives if code in courses]
        if len(options) > 1:
            by_semester.setdefault(f"{slot.year}-{slot.semester}", []).append({"key": slot.course.code, "options": options})
    return by_semester


def rebuild_inferred_courses(student: Student) -> None:
    """Replace the student's chart-inferred rows (source="pattern") for their current planned semester.

    Rows the student typed in (source="user") are never touched; if they typed a course the chart also names,
    their row wins. For "X or Y" slots, the student's pick (alternative_choices) is stored, else X.
    """
    StudentCourse.objects.filter(student=student, source="pattern").delete()
    if student.admission_year >= 2026:
        return  # no chart for the new curriculum (KnownGap gap_2026_no_structure)
    planning = (student.current_year, student.current_semester)
    typed = set(StudentCourse.objects.filter(student=student).values_list("course__code", flat=True))
    chosen_codes = set(student.alternative_choices.values())
    chosen = {course.code: course for course in Course.objects.filter(code__in=chosen_codes)}
    categories = category_map(student.programme)
    rows: dict[str, StudentCourse] = {}  # code -> row; why: a code can appear twice in a chart
    for slot in course_slots(student.programme):
        semester = (slot.year, slot.semester)
        course = chosen.get(student.alternative_choices.get(slot.course.code), slot.course)
        if semester > planning or course.code in typed:
            continue
        rows.setdefault(course.code, StudentCourse(
            student=student, course=course, pattern_slot=slot, source="pattern",
            status="completed" if semester < planning else "current",
            semester_taken=f"{slot.year}-{slot.semester}",
            category=categories.get(course.code, ""),
            # why: the chart's batch isn't stated (first-year courses changed over batches), and an unanswered
            # "X or Y" slot stores X by default; both are worth a second look
            needs_review=slot.year == 1 or (bool(slot.alternatives) and slot.course.code not in student.alternative_choices),
        ))
    StudentCourse.objects.bulk_create(rows.values())


def done_codes(student: Student, status: str | None = None) -> set[str]:
    """Codes the student has done or is doing this semester, widened to everything that stands in for them:
    the other side of an "X or Y" slot, timetable equivalents and hand-made code mappings (both directions).
    What the recommender must never suggest again. status: only "completed" or only "current" rows."""
    rows = StudentCourse.objects.filter(student=student).select_related("course", "pattern_slot__course")
    if status:
        rows = rows.filter(status=status)
    codes = {row.course.code for row in rows}
    for row in rows:
        if row.pattern_slot and row.pattern_slot.course:
            codes.add(row.pattern_slot.course.code)
            codes.update(row.pattern_slot.alternatives)
    return with_equivalents(codes)


def with_equivalents(codes: set[str]) -> set[str]:
    """The codes plus their timetable equivalents and hand-made code mappings, in both directions."""
    base = set(codes)
    codes = set(codes)
    codes.update(CourseEquivalent.objects.filter(course__code__in=base).values_list("equivalent_code", flat=True))
    codes.update(CourseEquivalent.objects.filter(equivalent_code__in=base).values_list("course__code", flat=True))
    codes.update(CodeMapping.objects.filter(to_course__code__in=base).values_list("from_code", flat=True))
    codes.update(CodeMapping.objects.filter(from_code__in=base).values_list("to_course__code", flat=True))
    return codes
