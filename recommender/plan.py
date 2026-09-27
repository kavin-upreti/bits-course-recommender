"""check_plan (todo.md section 7): do these courses fit with the student's current ones? Also used by
get_eligible_courses to drop courses that can't fit at all. The section search itself is in timetable.py."""
from catalog.models import Course, Offering, Rule

from . import config
from .codes import normalise_code, resolve_course
from .context import StudentContext, resolve_preferences
from .timetable import PlanOption, clash_detail, offerings_for, plan_options, search_sections


def offered(course: Course, semester_tag: str) -> bool:
    return Offering.objects.filter(course=course, semester_tag=semester_tag, sections__cancelled=False).exists()


def _plan_codes(ctx: StudentContext, courses: list[str]) -> tuple[list[Course], str | None]:
    """The courses to add (current ones dropped), or a problem sentence."""
    codes = list(dict.fromkeys(normalise_code(code) for code in courses))
    if not 1 <= len(codes) <= config.MAX_PLAN_COURSES:
        return [], f"Give between 1 and {config.MAX_PLAN_COURSES} course codes."
    found = []
    for code in codes:
        course = resolve_course(code)
        if course is None:
            return [], f"No course with code {code}"
        if not offered(course, ctx.semester_tag):
            return [], f"{course.code} isn't offered this semester"
        if course.code not in ctx.current and course not in found:
            found.append(course)
    return found, None


def _exam_conflicts(offerings: dict[str, Offering], notes: list[str]) -> list[dict]:
    """Midsem / compre at the same date and session; courses with no exam slot are noted and skipped."""
    conflicts = []
    for kind in ("midsem", "compre"):
        seen: dict[tuple[str, str], str] = {}
        for code, offering in offerings.items():
            when = (getattr(offering, f"{kind}_date"), getattr(offering, f"{kind}_session"))
            if not all(when):
                notes.append(f"{code}'s {kind} slot isn't in the timetable, so its {kind} clashes weren't checked")
            elif when in seen:
                conflicts.append({"a": seen[when], "b": code, "type": kind, "detail": f"both on {when[0]}, {when[1]}"})
            else:
                seen[when] = code
    return conflicts


def _units_conflict(ctx: StudentContext, added: list[Course], notes: list[str]) -> tuple[int, list[dict]]:
    """Total units (current courses except audit ones + the added ones) against the batch's maximum."""
    current = Course.objects.filter(code__in=ctx.current_courses)
    total = sum(course.units or 0 for course in current if ctx.category_map.get(course.code) != "AUDIT")
    total += sum(course.units or 0 for course in added)
    if ctx.max_units is None:
        notes.append("maximum units for your batch isn't in the data, so it wasn't checked")
        return total, []
    if total > ctx.max_units:
        return total, [{"a": "all courses", "b": "units", "type": "units",
                        "detail": f"total {total} units, maximum is {ctx.max_units}"}]
    return total, []


def _higher_degree_conflict(ctx: StudentContext, added: list[Course]) -> list[dict]:
    """More higher degree (G) courses than the regulations allow per semester (current ones count too)."""
    rule = Rule.objects.filter(rule_id="reg_higher_degree_course").first()
    limit = (rule.values or {}).get("max_per_semester") if rule else None
    if limit is None:
        return []
    current = Course.objects.filter(code__in=ctx.current_courses, is_higher_degree=True)
    codes = sorted({course.code for course in current} | {course.code for course in added if course.is_higher_degree})
    if len(codes) <= limit:
        return []
    return [{"a": ", ".join(codes), "b": "higher degree limit", "type": "higher_degree",
             "detail": f"{len(codes)} higher degree courses ({', '.join(codes)}), at most {limit} per semester"}]


def _problem_sentence(conflict: dict) -> str:
    a, b, detail = conflict["a"], conflict["b"], conflict["detail"]
    return {
        "midsem": f"{a} and {b} have their midsem at the same time ({detail}).",
        "compre": f"{a} and {b} have their compre at the same time ({detail}).",
        "units": f"Too many units: {detail}.",
        "higher_degree": f"Too many higher degree courses: {detail}.",
        "class": f"{a} clashes with {b} in every section combination (e.g. {detail}).",
        "lunch": f"{a} and {b} together always leave no lunch hour free (e.g. {detail}).",
    }[conflict["type"]]


def _failure(conflicts: list[dict], current: list[str], problem: str | None = None) -> dict:
    return {"ok": False, "problem": problem or _problem_sentence(conflicts[0]), "conflicts": conflicts,
            "includes_current_courses": current}


def _diagnose(added: list[str], offerings: dict[str, Offering]) -> list[dict]:
    """Pairs (an added course, any other scheduled course) that have no valid assignment on their own."""
    conflicts, checked = [], set()
    for a in added:
        for b in offerings:
            if a == b or frozenset((a, b)) in checked:
                continue
            checked.add(frozenset((a, b)))
            variables = [options for code in (a, b) for options in plan_options(offerings[code], False, None, config.EARLY_PERIODS)]
            result = search_sections(variables, config.LUNCH_PERIODS, config.PLAN_SEARCH_NODE_LIMIT)
            if result.picks is None and not result.limit_hit:
                if result.first_clash:
                    conflicts.append({"a": a, "b": b, "type": "class", "detail": clash_detail(result.first_clash, a)})
                else:
                    conflicts.append({"a": a, "b": b, "type": "lunch", "detail": result.first_lunch})
    return conflicts


def _preference_notes(picks: list[PlanOption], avoid_8am: bool, avoid_day: str | None) -> list[str]:
    """One note per pick that still breaks a preference, naming the course, the section and the problem."""
    notes = []
    for option in sorted(picks, key=lambda option: (option.code, option.type)):
        wants, has = [], []
        if avoid_8am and option.early_days:
            wants.append("avoids 8 AM")
            has.append(f"8 AM on {' '.join(option.early_days)}")
        if avoid_day and option.on_avoid_day:
            day = config.DAY_NAMES.get(avoid_day, avoid_day)
            wants.append(f"keeps {day} free")
            has.append(f"class on {day}")
        if wants:
            notes.append(f"no section combination {' or '.join(wants)} for {option.code}; chose {option.section_id} ({'; '.join(has)})")
        if option.untimed:
            notes.append(f"class times for {option.code} {option.section_id} aren't listed, so clashes with it couldn't be checked")
    return notes


def check_plan(ctx: StudentContext, courses: list[str], avoid_8am: bool | None = None,
               avoid_day: str | None = None) -> dict:
    """Do these courses fit with the student's current ones (exams, units, classes, lunch)? Picks sections."""
    added, problem = _plan_codes(ctx, courses)
    if problem:
        return {"ok": False, "problem": problem}
    settings = resolve_preferences(ctx, avoid_8am, avoid_day)
    notes: list[str] = []
    current_offerings = offerings_for(ctx.current_courses)
    for code in ctx.current_courses:
        if code not in current_offerings:
            notes.append(f"{code} (current) isn't in this semester's timetable, so it wasn't checked")
    offerings = {**current_offerings, **offerings_for([course.code for course in added])}
    current = [code for code in ctx.current_courses if code in current_offerings]
    added_codes = [course.code for course in added]

    total_units, unit_conflicts = _units_conflict(ctx, added, notes)
    conflicts = _higher_degree_conflict(ctx, added) + _exam_conflicts(offerings, notes) + unit_conflicts
    if conflicts:
        return _failure(conflicts, current)

    variables = [options for code in sorted(offerings)
                 for options in plan_options(offerings[code], settings["avoid_8am"], settings["avoid_day"], config.EARLY_PERIODS)]
    result = search_sections(variables, config.LUNCH_PERIODS, config.PLAN_SEARCH_NODE_LIMIT)
    if result.picks is None:
        if result.limit_hit:
            # ponytail: a safety net against a hung request; real checks use <= ~250 of the 200k steps (2026-09-28)
            return {**_failure([], current, "too many combinations to check; try fewer courses"), "limit_hit": True}
        conflicts = _diagnose(added_codes, offerings)
        return _failure(conflicts, current, None if conflicts else "Each pair of courses fits, but not all of them together.")
    if result.limit_hit:
        notes.append("search limit reached; this is a valid timetable but maybe not the best one")
    sections: dict[str, dict[str, str]] = {code: {} for code in added_codes}
    for option in result.picks:
        if option.code in sections:
            sections[option.code][option.type] = option.section_id
    result_dict = {"ok": True, "sections": {code: dict(sorted(picked.items())) for code, picked in sections.items()},
                   "includes_current_courses": current, "total_units": total_units,
                   "notes": notes + _preference_notes(result.picks, settings["avoid_8am"], settings["avoid_day"])}
    # ponytail: the search minimises total violations, so a tie between a current course's 8 AM and the new
    # course's could land on the new one; exhaustive per-course checks if that shows up in practice
    missed = sorted({option.code for option in result.picks if option.code in sections and option.violations})
    if missed:
        result_dict["missed_preferences"] = missed  # given courses that fit only by breaking 8 AM / free day
    return result_dict
