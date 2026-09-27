"""Clash-free section picks for a semester's courses, and their exam calendar.

The loaded timetable is treated as the timetable of whatever semester the student is planning.
Sections of one course and type with identical timings are interchangeable for clashes, so they are one
`Choice` ("+2" = two more sections at the same times); that shrinks the search a lot (MATH F211: 12 tutorials).
"""
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from catalog.models import Offering, Rule, Section

DAYS = ("M", "T", "W", "Th", "F", "S")
MIDSEM_ORDER = ("FN1", "FN2", "AN1", "AN2")
COMPRE_ORDER = ("FN", "AN")
MAX_SOLUTIONS = 5000  # ponytail: first 5000 found are ranked; exhaustive ranking if the top ones look poor


@dataclass
class Choice:
    """One pick for a (course, section type): every section here meets at the same times."""

    code: str
    type: str
    sections: list[Section]
    slots: frozenset[tuple[str, int]]  # {("M", 1), ("M", 2)}

    @property
    def section(self) -> Section:
        return self.sections[0]

    @property
    def others(self) -> int:
        return len(self.sections) - 1


@dataclass
class Timetable:
    picks: list[Choice]
    by_slot: dict[tuple[str, int], Choice] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_slot = {slot: pick for pick in self.picks for slot in pick.slots}

    def score(self) -> tuple[int, int, int]:
        """Lower is better: idle hours between classes, 8 AM classes, days with any class."""
        gaps = 0
        for day in DAYS:
            periods = sorted(period for d, period in self.by_slot if d == day)
            if periods:
                gaps += periods[-1] - periods[0] + 1 - len(periods)
        eight_ams = sum(period == 1 for _, period in self.by_slot)
        days = len({day for day, _ in self.by_slot})
        return gaps, eight_ams, days

    def teaches(self, code: str, name: str) -> bool:
        """Does some section this timetable can use for `code` have `name` as an instructor?"""
        return any(name in section.instructors for pick in self.picks if pick.code == code for section in pick.sections)


def offerings_for(codes: list[str]) -> dict[str, Offering]:
    """code -> its offering in the loaded timetable. ponytail: a code with two com codes uses the first."""
    offerings: dict[str, Offering] = {}
    for offering in Offering.objects.filter(course__code__in=codes).select_related("course").prefetch_related("sections").order_by("pk"):
        offerings.setdefault(offering.course.code, offering)
    return offerings


def choices_for(offering: Offering) -> list[list[Choice]]:
    """One list per section type the course has (lecture / tutorial / practical); a timetable picks one from each."""
    by_type: dict[str, dict[frozenset, Choice]] = {}
    for section in offering.sections.all():
        if section.cancelled:
            continue
        slots = frozenset((day, period) for day, periods in section.timings.items() for period in periods)
        groups = by_type.setdefault(section.type, {})
        if slots in groups:
            groups[slots].sections.append(section)
        else:
            groups[slots] = Choice(offering.course.code, section.type, [section], slots)
    return [list(groups.values()) for groups in by_type.values()]


@dataclass
class TimetableFilters:
    """What the student asked for above the timetables. Each one is a sort key (fewest breaks first), not a hard
    filter, so there is always something to show."""

    no_8am: bool = False
    compact: bool = False
    free_day: str = ""
    teachers: list[tuple[str, str]] = field(default_factory=list)  # (course code, instructor)
    avoid: set[tuple[str, int]] = field(default_factory=set)       # (day, period) to keep free

    def key(self, timetable: "Timetable") -> tuple:
        """Missing teachers, then classes at avoided times, 8 AMs, classes on the free day, gaps; then the default."""
        gaps, eight_ams, _ = timetable.score()
        parts = [sum(not timetable.teaches(code, name) for code, name in self.teachers),
                 sum(slot in self.avoid for slot in timetable.by_slot)]
        parts += [eight_ams if self.no_8am else 0, sum(day == self.free_day for day, _ in timetable.by_slot),
                  gaps if self.compact else 0]
        return (*parts, *timetable.score())

    def prefer_teachers(self, choices: list["Choice"]) -> None:
        """Show the asked-for teacher's section first where a group has several at the same time."""
        wanted = {(code, name) for code, name in self.teachers}
        for choice in choices:
            choice.sections.sort(key=lambda section: not any((choice.code, name) in wanted for name in section.instructors))


def generate(offerings: list[Offering], limit: int = 20, order=None) -> list[Timetable]:
    """The `limit` best clash-free timetables that keep a lunch hour free every day, or [] if there are none.
    Best = lowest `order(timetable)` (default: `Timetable.score`)."""
    variables = [options for offering in offerings for options in choices_for(offering)]
    variables.sort(key=len)  # why: fewest options first, so dead ends are found early
    found: list[Timetable] = []

    # why a dict, not a set: a course's own sections may share a slot (BITS F102 lists its lecture and tutorial
    # at the same hour), which isn't a clash; only two different courses in one slot are
    taken: dict[tuple[str, int], str] = {}
    lunch = Rule.objects.get(rule_id="tt_lunch_hour").values["lunch_periods"]

    def search(index: int, picks: list[Choice]) -> None:
        if len(found) >= MAX_SOLUTIONS:
            return
        if index == len(variables):
            found.append(Timetable(list(picks)))
            return
        for option in variables[index]:
            if any(taken.get(slot, option.code) != option.code for slot in option.slots):
                continue
            added = [slot for slot in option.slots if slot not in taken]
            taken.update(dict.fromkeys(added, option.code))
            # tt_lunch_hour: every day keeps one of the lunch periods free
            if all(any((day, period) not in taken for period in lunch) for day in {day for day, _ in added}):
                picks.append(option)
                search(index + 1, picks)
                picks.pop()
            for slot in added:
                del taken[slot]

    search(0, [])
    return sorted(found, key=order or Timetable.score)[:limit]


def always_clashing(offerings: list[Offering]) -> list[tuple[str, str]]:
    """Pairs of courses no section pick can separate; explains an empty `generate` result."""
    options = {offering.course.code: choices_for(offering) for offering in offerings}
    pairs = []
    codes = list(options)
    for i, first in enumerate(codes):
        for second in codes[i + 1:]:
            if any(all(a.slots & b.slots for a in var_a for b in var_b) for var_a in options[first] for var_b in options[second]):
                pairs.append((first, second))
    return pairs


def period_times() -> dict[int, str]:
    """Period -> "8:00" from the timetable legend; periods past the legend's last one continue hourly."""
    values = Rule.objects.get(rule_id="tt_periods").values
    times = {row["period"]: row["start"].lstrip("0") for row in values}
    last = max(times)
    for period in range(last + 1, 13):
        times[period] = f"{int(times[last].split(':')[0]) + period - last}:00"
    return times


def week_rows(timetable: Timetable, last_period: int) -> list[dict]:
    """Table rows for the week grid: one per period, a cell per day. A pick spanning consecutive periods
    becomes one cell with a rowspan; the periods it covers are marked `skip`."""
    times = period_times()
    rows = []
    for period in range(1, last_period + 1):
        cells = []
        for day in DAYS:
            pick = timetable.by_slot.get((day, period))
            if pick and timetable.by_slot.get((day, period - 1)) is pick:
                cells.append({"skip": True})
                continue
            span = 1
            while pick and timetable.by_slot.get((day, period + span)) is pick:
                span += 1
            cells.append({"pick": pick, "rowspan": span})
        rows.append({"time": f"{times[period]}–{times.get(period + 1, '')}".rstrip("–"), "start": times[period], "cells": cells})
    return rows


def week_dates(today: date) -> list[date]:
    """Mon..Sat of the current week; the week rolls over at the end of Sunday."""
    monday = today - timedelta(days=today.weekday())
    return [monday + timedelta(days=offset) for offset in range(len(DAYS))]


def exam_date(text: str, semester_tag: str) -> date | None:
    """'10/10' -> date. The year comes from the tag ("2026-27 Sem 1"): Jul-Dec is the first year, Jan-Jun the second."""
    try:
        day, month = map(int, text.split("/"))
        first_year = int(semester_tag[:4])
        return date(first_year if month >= 7 else first_year + 1, month, day)
    except (ValueError, IndexError):
        return None


@dataclass
class ExamCalendar:
    kind: str  # Midsem / Compre
    sessions: list[tuple[str, str]]  # (code, "9:00–10:30")
    dates: list[date]
    grid: list[dict]  # [{"session": ..., "time": ..., "cells": [[code, ...] per date]}]
    warnings: list[str]
    undated: list[str]


def exam_calendar(offerings: list[Offering], kind: str) -> ExamCalendar:
    """Exam grid over the span of the student's exam dates (Sundays left out), with clash / back-to-back warnings."""
    order, rule_id, date_field, session_field = (
        (MIDSEM_ORDER, "tt_midsem_sessions", "midsem_date", "midsem_session") if kind == "Midsem"
        else (COMPRE_ORDER, "tt_compre_sessions", "compre_date", "compre_session")
    )
    times = Rule.objects.get(rule_id=rule_id).values
    exams: dict[tuple[date, str], list[str]] = {}
    undated = []
    for offering in offerings:
        when = exam_date(getattr(offering, date_field), offering.semester_tag)
        session = getattr(offering, session_field)
        if when is None or session not in order:
            undated.append(offering.course.code)
            continue
        exams.setdefault((when, session), []).append(offering.course.code)

    dates: list[date] = []
    if exams:
        first, last = min(day for day, _ in exams), max(day for day, _ in exams)
        dates = [first + timedelta(days=n) for n in range((last - first).days + 1) if (first + timedelta(days=n)).weekday() != 6]
    fmt = lambda clock: clock.lstrip("0")  # noqa: E731
    sessions = [(code, f"{fmt(times[code]['start'])}–{fmt(times[code]['end'])}") for code in order]
    grid = [{"session": code, "time": time, "cells": [exams.get((day, code), []) for day in dates]} for code, time in sessions]

    warnings = []
    for (day, session), codes in sorted(exams.items()):
        label = day.strftime("%-d %b")
        if len(codes) > 1:
            warnings.append(f"{' and '.join(codes)} are at the same time ({label}, {session}).")
        following = order.index(session) + 1
        if following < len(order) and (day, order[following]) in exams:
            warnings.append(f"{', '.join(codes)} and {', '.join(exams[(day, order[following])])} ({label}) are back-to-back.")
    return ExamCalendar(kind, sessions, dates, grid, warnings, undated)


def timing_text(timings: dict[str, list[int]], times: dict[int, str]) -> str:
    """{"T": [3], "Th": [3], "F": [3]} -> "T Th F 10:00–10:50"; days meeting at different hours are listed apart."""
    by_periods: dict[tuple[int, ...], list[str]] = {}
    for day in DAYS:
        if timings.get(day):
            by_periods.setdefault(tuple(sorted(timings[day])), []).append(day)
    parts = []
    for periods, days in by_periods.items():
        end = times.get(periods[-1], "?").split(":")[0] + ":50"  # why: every period is 50 minutes (tt_periods)
        parts.append(f"{' '.join(days)} {times.get(periods[0], '?')}–{end}")
    return ", ".join(parts)


def timetable_term() -> tuple[int, int] | None:
    """(first calendar year, semester) of the loaded timetable: "2026-27 Sem 1" -> (2026, 1). None if none is loaded."""
    tag = Offering.objects.values_list("semester_tag", flat=True).first() or ""
    match = re.match(r"(\d{4})-\d{2} Sem (\d)", tag)
    return (int(match[1]), int(match[2])) if match else None


def semester_for(admission_year: int) -> tuple[int, int] | None:
    """The year-semester a batch is going into, per the loaded timetable: batch 2025 in 2026-27 Sem 1 -> (2, 1).
    ponytail: assumes the normal track; a repeated semester or a break would need a manual override."""
    term = timetable_term()
    return (term[0] - admission_year + 1, term[1]) if term else None


# ---------------------------------------------------------------- check_plan's section search (todo.md 7.3)

@dataclass
class PlanOption:
    """One pick for a (course, section type): the lowest-id section of a group meeting at identical times."""

    code: str
    type: str
    section_id: str
    slots: frozenset[tuple[str, int]]
    violations: int          # timetable preferences it breaks (8 AM, the day to keep free)
    early_days: list[str]    # days it has an 8 AM class
    on_avoid_day: bool
    untimed: bool            # no timings listed: occupies nothing, so clashes can't be checked


@dataclass
class PlanSearch:
    """The best assignment found (None = no valid one), and what the search ran into."""

    picks: list[PlanOption] | None
    violations: int
    nodes: int
    limit_hit: bool
    first_clash: tuple | None    # (code, section, other code, other section, day, period) of the first clash met
    first_lunch: str | None      # "together they fill periods 4, 5 and 6 on W"


def clash_detail(clash: tuple, first_code: str) -> str:
    """"both L1 on T, period 3" / "L1 and P2 on Th, period 8", with first_code's section named first."""
    code, section, other_code, other_section, day, period = clash
    if code != first_code:
        section, other_section = other_section, section
    pair = f"both {section}" if section == other_section else f"{section} and {other_section}"
    return f"{pair} on {day}, period {period}"


def section_order(section_id: str) -> tuple[str, int, str]:
    """Natural order: L2 before L10."""
    match = re.match(r"([A-Za-z]*)(\d*)(.*)", section_id)
    return match[1], int(match[2] or 0), match[3]


def plan_options(offering: Offering, avoid_8am: bool, avoid_day: str | None,
                 early_periods: set[int]) -> list[list[PlanOption]]:
    """One option list per required section type (types with a non-cancelled section)."""
    groups: dict[tuple[str, frozenset], list[Section]] = {}
    for section in offering.sections.all():
        if not section.cancelled:
            slots = frozenset((day, period) for day, periods in section.timings.items() for period in periods)
            groups.setdefault((section.type, slots), []).append(section)
    by_type: dict[str, list[PlanOption]] = {}
    for (kind, slots), sections in groups.items():
        first = min(sections, key=lambda section: section_order(section.section_id))
        early = sorted({day for day, period in slots if period in early_periods}, key=DAYS.index)
        on_day = bool(avoid_day) and any(day == avoid_day for day, _ in slots)
        violations = int(avoid_8am and bool(early)) + int(on_day)
        by_type.setdefault(kind, []).append(PlanOption(offering.course.code, kind, first.section_id, slots, violations,
                                                       early, on_day, not slots))
    return [sorted(options, key=lambda option: (option.violations, section_order(option.section_id)))
            for _, options in sorted(by_type.items())]


def search_sections(variables: list[list[PlanOption]], lunch_periods: set[int], node_limit: int) -> PlanSearch:
    """Backtracking with branch-and-bound: the assignment with the fewest preference violations, stopping at the
    first one with none. A course's own sections may share a slot (not a clash); two courses may not; every day
    keeps one lunch period free."""
    variables = sorted(variables, key=lambda options: (len(options), options[0].code, options[0].type))
    taken: dict[tuple[str, int], PlanOption] = {}
    picks: list[PlanOption] = []
    state = {"best": None, "best_violations": float("inf"), "nodes": 0, "limit": False, "clash": None, "lunch": None}

    def clash_with(option: PlanOption) -> tuple[str, int] | None:
        return next((slot for slot in sorted(option.slots) if slot in taken and taken[slot].code != option.code), None)

    def lunch_broken(option: PlanOption) -> str | None:
        for day in sorted({day for day, _ in option.slots}, key=DAYS.index):
            if all((day, period) in taken for period in lunch_periods):
                periods = sorted(lunch_periods)
                return f"together they fill periods {', '.join(map(str, periods[:-1]))} and {periods[-1]} on {day}"
        return None

    def search(index: int, violations: int) -> bool:
        """True = stop the whole search (a perfect assignment, or the node limit)."""
        state["nodes"] += 1
        if state["nodes"] > node_limit:
            state["limit"] = True
            return True
        if violations >= state["best_violations"]:
            return False
        if index == len(variables):
            state["best"], state["best_violations"] = list(picks), violations
            return violations == 0
        for option in variables[index]:
            slot = clash_with(option)
            if slot:
                other = taken[slot]
                state["clash"] = state["clash"] or (option.code, option.section_id, other.code, other.section_id, *slot)
                continue
            added = [slot for slot in option.slots if slot not in taken]
            taken.update(dict.fromkeys(added, option))
            broken = lunch_broken(option)
            if broken:
                state["lunch"] = state["lunch"] or broken
            else:
                picks.append(option)
                stop = search(index + 1, violations + option.violations)
                picks.pop()
                if stop:
                    for slot in added:
                        del taken[slot]
                    return True
            for slot in added:
                del taken[slot]
        return False

    search(0, 0)
    best = state["best"]
    return PlanSearch(best, 0 if best is None else int(state["best_violations"]), state["nodes"], state["limit"],
                      state["clash"], state["lunch"])
