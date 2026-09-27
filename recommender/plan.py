"""check_plan (todo.md section 7): do these courses fit with the student's current ones? Also used by
get_eligible_courses to drop courses that can't fit at all. The section search is timetable.generate, the same one
the timetable page uses."""
from catalog.models import Course, Offering, Rule

from . import config
from .codes import normalise_code, resolve_course
from .context import StudentContext, resolve_preferences
from .timetable import DAYS, Choice, Timetable, TimetableFilters, always_clash, generate, lunch_periods, offerings_for


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


def _lunch_detail(timetable: Timetable) -> str:
    """"together they fill periods 4, 5 and 6 on M": the first day a timetable (made without the lunch rule) fills."""
    periods = sorted(lunch_periods())
    day = next(day for day in DAYS if all((day, period) in timetable.by_slot for period in periods))
    return f"together they fill periods {', '.join(map(str, periods[:-1]))} and {periods[-1]} on {day}"


def _diagnose(added: list[str], offerings: dict[str, Offering]) -> list[dict]:
    """Pairs (an added course, any other scheduled course) that have no timetable on their own, and why."""
    conflicts, checked = [], set()
    for a in added:
        for b in offerings:
            if a == b or frozenset((a, b)) in checked:
                continue
            checked.add(frozenset((a, b)))
            pair = [offerings[a], offerings[b]]
            if generate(pair, limit=1):
                continue
            without_lunch = generate(pair, limit=1, lunch=False)
            if without_lunch:
                conflicts.append({"a": a, "b": b, "type": "lunch", "detail": _lunch_detail(without_lunch[0])})
            else:
                conflicts.append({"a": a, "b": b, "type": "class",
                                  "detail": always_clash(*pair) or "no pick of their sections avoids a clash"})
    return conflicts


def _broken(pick: Choice, avoid_8am: bool, avoid_day: str | None) -> tuple[list[str], bool]:
    """(days the pick has an 8 AM class, if avoiding them; does it meet on the day to keep free)."""
    early = sorted({day for day, period in pick.slots if period in config.EARLY_PERIODS}, key=DAYS.index) if avoid_8am else []
    return early, bool(avoid_day) and any(day == avoid_day for day, _ in pick.slots)


def _preference_notes(picks: list[Choice], avoid_8am: bool, avoid_day: str | None) -> list[str]:
    """One note per pick that still breaks a preference, naming the course, the section and the problem."""
    notes = []
    for pick in sorted(picks, key=lambda pick: (pick.code, pick.type)):
        early, on_day = _broken(pick, avoid_8am, avoid_day)
        wants, has = [], []
        if early:
            wants.append("avoids 8 AM")
            has.append(f"8 AM on {' '.join(early)}")
        if on_day:
            day = config.DAY_NAMES.get(avoid_day, avoid_day)
            wants.append(f"keeps {day} free")
            has.append(f"class on {day}")
        section = pick.section.section_id
        if wants:
            notes.append(f"no section combination {' or '.join(wants)} for {pick.code}; chose {section} ({'; '.join(has)})")
        if not pick.slots:
            notes.append(f"class times for {pick.code} {section} aren't listed, so clashes with it couldn't be checked")
    return notes


def check_plan(ctx: StudentContext, courses: list[str], avoid_8am: bool | None = None,
               avoid_day: str | None = None) -> dict:
    """Do these courses fit with the student's current ones (exams, units, classes, lunch)? Picks sections: the best
    timetable by the timetable page's own order (fewest 8 AMs / classes on the free day first, when asked)."""
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

    order = TimetableFilters(no_8am=settings["avoid_8am"], free_day=settings["avoid_day"] or "").key
    found = generate([offerings[code] for code in sorted(offerings)], limit=1, order=order)
    if not found:
        conflicts = _diagnose(added_codes, offerings)
        return _failure(conflicts, current, None if conflicts else "Each pair of courses fits, but not all of them together.")
    picks = found[0].picks
    sections: dict[str, dict[str, str]] = {code: {} for code in added_codes}
    for pick in picks:
        if pick.code in sections:
            sections[pick.code][pick.type] = pick.section.section_id
    result = {"ok": True, "sections": {code: dict(sorted(picked.items())) for code, picked in sections.items()},
              "includes_current_courses": current, "total_units": total_units,
              "notes": notes + _preference_notes(picks, settings["avoid_8am"], settings["avoid_day"])}
    missed = sorted({pick.code for pick in picks if pick.code in sections and any(_broken(pick, settings["avoid_8am"], settings["avoid_day"]))})
    if missed:
        result["missed_preferences"] = missed  # given courses that fit only by breaking 8 AM / free day
    return result
