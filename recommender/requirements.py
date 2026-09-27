"""What's left to finish (ideation 6.4, first cut): elective counts per category and minor progress.

Counts use the courses done *and* this semester's compulsory ones. Electives are allocated greedily, in the
regulation's order: DEL up to its requirement, then HUEL up to its requirement, anything beyond overflows into OPEL.
Project-course limits and the minor-overlap caps are not applied yet.
"""
from dataclasses import dataclass, field

from catalog.models import CategoryRequirement, Course, MinorCourse, Programme, Rule
from students.models import Student, StudentCourse

from .categories import category_map
from .history import course_slots, done_codes

COMPULSORY = ("GIR", "CHART", "CDC", "AUDIT")


@dataclass
class Need:
    """One elective requirement, in courses and units."""

    category: str  # DEL / HUEL / OPEL
    label: str
    required_courses: int
    required_units: int
    done_courses: int
    done_units: int
    note: str = ""
    degree: str = ""  # dual degrees: which degree this DEL count belongs to

    @property
    def courses_left(self) -> int:
        return max(0, self.required_courses - self.done_courses)

    @property
    def units_left(self) -> int:
        return max(0, self.required_units - self.done_units)


@dataclass
class MinorProgress:
    name: str
    core_left: list[Course] = field(default_factory=list)
    electives_left: int = 0
    done_courses: int = 0
    required_courses: int = 0


def degrees(programme: Programme) -> list[Programme]:
    """The single degrees a programme is made of: itself, or a dual degree's two components."""
    if programme.type == "dual" and programme.first_component and programme.second_component:
        return [programme.first_component, programme.second_component]
    return [programme]


def _requirement(**lookup) -> CategoryRequirement:
    return CategoryRequirement.objects.get(**lookup)


def _units(course: Course) -> int:
    return course.units or 0


# From students' degree audits (A7 and B5A8, 2026-09-26): coursework is 44 courses / 129 units and the GIR sub-total
# 21 courses / 54 units. The Bulletin edition we have predates the 6-course Science Foundation (it says 41 / 51).
COURSEWORK_TOTAL = (44, 129)
GIR_TOTAL = (21, 54)


def opel_requirement(degree: Programme, categories: dict[str, str]) -> tuple[int, int]:
    """(courses, units) of open electives for a single degree: what's left of the coursework total after the GIRs,
    the discipline core and the DELs, never below the Bulletin's minimum (5 courses / 15 units).
    Matches the A7 audit: 44 - 21 - 14 - 4 = 5 courses, 129 - 54 - 48 - 12 = 15 units."""
    opel = _requirement(code="OPEL")
    cdc = [code for code, category in categories.items() if category == "CDC"]
    core_courses = degree.core_courses or len(cdc)
    core_units = degree.core_units or sum(_units(course) for course in Course.objects.filter(code__in=cdc))
    courses = COURSEWORK_TOTAL[0] - GIR_TOTAL[0] - core_courses - (degree.del_courses or 0)
    units = COURSEWORK_TOTAL[1] - GIR_TOTAL[1] - core_units - (degree.del_units or 0)
    return max(opel.min_courses, courses), max(opel.min_units, units)


def elective_needs(student: Student, typed_codes: list[str] | None = None) -> list[Need]:
    """HUEL once; DEL per degree (a dual degree meets each degree's discipline requirements separately); OPEL for a
    single degree only: a dual degree's audit has no open-elective requirement at all.

    typed_codes replaces the student's saved electives: the live preview while they edit them."""
    rows = StudentCourse.objects.filter(student=student).select_related("course")
    if typed_codes is not None:
        rows = rows.filter(source="pattern")
    by_code = {row.course.code: row.course for row in rows}
    if typed_codes is not None:
        by_code.update({course.code: course for course in Course.objects.filter(code__in=typed_codes)})
    courses = sorted(by_code.values(), key=lambda course: course.code)
    huel = _requirement(code="HUEL")
    student_categories = category_map(student.programme)
    huel_used = [course for course in courses if student_categories.get(course.code) == "HUEL"][: huel.min_courses]
    needs = [Need("HUEL", "HUEL", huel.min_courses, huel.min_units,
                  len(huel_used), sum(map(_units, huel_used)))]

    parts = degrees(student.programme)
    for degree in parts:
        categories = category_map(degree)
        degree_name = degree.name if len(parts) > 1 else ""
        dels = [course for course in courses if categories.get(course.code) == "DEL"]
        del_used = dels[: degree.del_courses or 0]
        note = "" if degree.del_courses else "The Bulletin gives no DEL count for this programme."
        needs.append(Need("DEL", "DEL", degree.del_courses or 0, degree.del_units or 0,
                          len(del_used), sum(map(_units, del_used)), note, degree_name))

    if len(parts) > 1:
        # why: a dual degree's audit lists GIR, both cores and both DEL pools, and no open requirement (B5A8 audit,
        # 2026-09-26); the dual charts have no OPEL slots either
        return needs
    # single degree: `categories`, `del_used` are the (only) degree's from the loop above
    used = {course.code for course in del_used + huel_used}
    opel_done = [course for course in courses if categories.get(course.code) not in COMPULSORY and course.code not in used]
    opel_courses, opel_units = opel_requirement(student.programme, categories)
    needs.append(Need("OPEL", "OPEL", opel_courses, opel_units, len(opel_done), sum(map(_units, opel_done))))
    return needs


def minor_progress(student: Student) -> MinorProgress | None:
    """Minor core courses not yet done, and how many minor electives are still needed (minor_rules).
    Every core course is treated as compulsory for the minor."""
    if student.minor is None:
        return None
    rules = Rule.objects.get(rule_id="minor_units").values
    covered = done_codes(student)
    links = MinorCourse.objects.filter(minor=student.minor).select_related("course").order_by("course__code")
    core = [link.course for link in links if link.role == "core"]
    electives_done = sum(link.course.code in covered for link in links if link.role == "elective")
    core_left = [course for course in core if course.code not in covered]
    core_done = len(core) - len(core_left)
    required = student.minor.min_courses or rules["total_min_courses"]
    electives_left = max(
        rules["electives_min_courses"] - electives_done,
        required - core_done - electives_done - len(core_left),
        0,
    )
    return MinorProgress(student.minor.name, core_left, electives_left, core_done + electives_done, required)


def compulsory_left(student: Student) -> int:
    """Chart courses still ahead, after the semester being planned (PS / thesis not counted)."""
    planning = (student.current_year, student.current_semester)
    return len({slot.course.code for slot in course_slots(student.programme) if (slot.year, slot.semester) > planning})



@dataclass
class SemesterLoad:
    """Units already fixed for the planned semester (its compulsory courses) against the per-semester cap."""

    fixed_units: int
    max_units: int | None  # None = not stated for this batch (reg_max_units_2026)

    @property
    def free_units(self) -> int | None:
        return None if self.max_units is None else self.max_units - self.fixed_units

    @property
    def over(self) -> bool:
        return self.max_units is not None and self.fixed_units > self.max_units


def semester_load(student: Student) -> SemesterLoad:
    """25 units max for first-degree students up to the 2025 batch (Regulations); the 2026 credit-hour framework's
    cap isn't in our documents, so it stays None. Audit courses don't count towards the cap."""
    rule_id = "reg_max_units_2026" if student.admission_year >= 2026 else "reg_max_units"
    cap = Rule.objects.get(rule_id=rule_id).values.get("first_degree_max_units")
    current = StudentCourse.objects.filter(student=student, status="current").exclude(category="AUDIT").select_related("course")
    return SemesterLoad(sum(_units(row.course) for row in current), cap)
