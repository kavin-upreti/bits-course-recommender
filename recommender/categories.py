"""Per-programme course categories (ideation Part 6.3, first cut): what each course is *for this programme*.

The same course is a CDC for one programme and an OPEL for another, so this is always computed per programme.
Dual-degree layer, minor tags and the "not classifiable" flag come later with the full `classify`.
"""
from catalog.models import AuditCourse, Course, GirCourse, HuelPoolCourse, PatternSlot, Programme, ProgrammeCourse

ELECTIVE_CATEGORIES = ("DEL", "HUEL", "OPEL")
# The Bulletin's "Mathematics Foundation" GIR heading names no courses; students' degree audits list it as these
# 4 courses / 12 units (checked on an A7 and a B5A8 audit, 2026-09-26).
MATHS_FOUNDATION = {"MATH F101", "MATH F102", "MATH F113", "MATH F211"}


def _with_alternatives(links) -> set[str]:
    """Codes of the linked courses plus every code the Bulletin allows in their place ("CS F211 or BITS F232")."""
    codes: set[str] = set()
    for course_code, alternatives in links:
        codes.add(course_code)
        codes.update(alternatives)
    return codes


def own_disciplines(programme: Programme) -> set[str]:
    """Course prefixes of the student's own degree(s); a dual degree has two."""
    parts = [programme.first_component, programme.second_component] if programme.type == "dual" else [programme]
    return {part.discipline_code for part in parts if part and part.discipline_code}


def chart_slots(programme: Programme, slot_type: str = "named") -> list[PatternSlot]:
    """The programme's chart slots of one type (named courses, or elective placeholders). A dual chart starts at
    year 2 ("Same as First degree Programme"), so its missing years come from the first degree's chart."""
    slots = list(PatternSlot.objects.filter(programme=programme, slot_type=slot_type).select_related("course"))
    if programme.type == "dual" and programme.first_component:
        covered_years = {slot.year for slot in slots}
        slots += [
            slot for slot in PatternSlot.objects.filter(programme=programme.first_component, slot_type=slot_type).select_related("course")
            if slot.year not in covered_years
        ]
    return sorted(slots, key=lambda slot: (slot.year, slot.semester, slot.pk))


def category_map(programme: Programme) -> dict[str, str]:
    """code -> AUDIT / GIR / CDC / CHART / DEL / HUEL / OPEL for every course in the catalog. First match wins,
    in the order of ideation 6.3.

    CHART = named in the programme's chart but in neither the GIR nor the CDC list (a long tail: code-mapping
    leftovers, programme-specific foundation courses). Compulsory, so never an elective; which requirement it
    counts towards is left open rather than guessed.
    """
    links = ProgrammeCourse.objects.filter(programme=programme)
    cdc = _with_alternatives(links.filter(category="CDC").values_list("course__code", "alternative_group"))
    dels = _with_alternatives(links.exclude(category="CDC").values_list("course__code", "alternative_group"))
    gir = _with_alternatives(GirCourse.objects.values_list("course__code", "alternative_group")) | MATHS_FOUNDATION
    audit = set(AuditCourse.objects.values_list("code", flat=True))
    huel = set(HuelPoolCourse.objects.values_list("course__code", flat=True))
    chart = {slot.course.code for slot in chart_slots(programme) if slot.course}
    disciplines = own_disciplines(programme)

    categories: dict[str, str] = {}
    for code, department in Course.objects.values_list("code", "department"):
        if code in audit:
            categories[code] = "AUDIT"
        elif code in gir:
            categories[code] = "GIR"
        elif code in cdc:
            categories[code] = "CDC"
        elif code in chart:
            categories[code] = "CHART"
        elif code in dels:
            categories[code] = "DEL"
        elif code in huel and department not in disciplines:  # huel_rules: own discipline can't count as HUEL
            categories[code] = "HUEL"
        else:
            categories[code] = "OPEL"
    return categories


def elective_choices(programme: Programme, admission_year: int) -> dict[str, list[Course]]:
    """Courses a student of this programme could have taken as each elective category, for the profile form.

    Compulsory courses (CDC, GIR, CHART) and audit courses are never offered here. 2026-only courses are hidden
    from earlier batches.
    """
    categories = category_map(programme)
    courses = Course.objects.order_by("code")
    if admission_year < 2026:
        courses = courses.filter(only_2026_batch=False)
    choices: dict[str, list[Course]] = {category: [] for category in ELECTIVE_CATEGORIES}
    for course in courses:
        category = categories.get(course.code)
        if category in choices:
            choices[category].append(course)
    return choices
