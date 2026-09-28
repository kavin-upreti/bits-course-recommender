"""Register -> profile -> past electives -> home (profile summary). Recommendations come later."""
import re
from difflib import SequenceMatcher

from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import Max
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from catalog.models import Course, KnownGap, PatternSlot, Programme
from recommender.config import DAY_NAMES
from recommender.context import requirement_status
from recommender.categories import category_map, chart_slots, elective_choices
from recommender.history import alternative_slots, course_slots, rebuild_inferred_courses
from recommender.requirements import compulsory_left, elective_needs, minor_progress, semester_load
from recommender.timetable import (
    DAYS, TimetableFilters, always_clashing, exam_calendar, exam_date, generate, offerings_for, period_times, semester_for, timing_text,
    week_dates, week_rows,
)

from .bits_id import BRANCH_CODES, BitsIdError, ParsedId, parse_bits_id, resolve_programme, second_degree_options
from .forms import MINOR_FROM_YEAR, ElectiveFormSet, ProfileForm, RegisterForm
from .models import Student, StudentCourse
from .templatetags.text import course_title

FIRST_ELECTIVE_SEMESTER = (2, 1)  # electives start in 2-1; earlier semesters are all compulsory


def register(request: HttpRequest) -> HttpResponse:
    form = RegisterForm(request.POST if request.method == "POST" else None)
    if request.method == "POST" and form.is_valid():
        login(request, form.save())
        return redirect("profile")
    return render(request, "students/register.html", {"form": form})


def max_year(programme: Programme) -> int:
    """Length of the programme in years, from its chart (4 for single degrees, 5 for dual)."""
    return PatternSlot.objects.filter(programme=programme).aggregate(Max("year"))["year__max"] or 4


def past_elective_semesters(student: Student) -> list[str]:
    """Completed semesters from 2-1 up to (not including) the one being planned, e.g. ["2-1", "2-2"]."""
    planning = (student.current_year, student.current_semester)
    return [
        f"{year}-{sem}" for year in range(1, student.current_year + 1) for sem in (1, 2)
        if FIRST_ELECTIVE_SEMESTER <= (year, sem) < planning
    ]


def id_parts(bits_id: str, parsed: ParsedId) -> list[tuple[str, str, bool]]:
    """(text, caption, is_branch) for each segment of the ID, e.g. ("A7", "B.E. Computer Science", True)."""
    codes = [code for code in (parsed.first_code, parsed.second_code) if code]
    track_start = 4 + 2 * len(codes)  # "PS" (practice school) or "TS" (thesis), then the 4-digit number
    track = bits_id[track_start:track_start + 2]
    return [
        (bits_id[:4], "batch", False),
        *[(code, BRANCH_CODES[code], True) for code in codes],
        (track, "practice school" if track == "PS" else "thesis", False),
        (bits_id[track_start + 2:-1], "your number", False),
        (bits_id[-1], f"{parsed.campus} campus", False),
    ]


def read_alternative_choices(post, slots_by_semester: dict[str, list[dict]], semesters: list[str]) -> dict[str, str]:
    """{"ECON F211": "MGTS F211"} from the "alt:<code>" dropdowns of the given semesters; unknown codes are ignored."""
    picked = {}
    for semester in semesters:
        for slot in slots_by_semester.get(semester, []):
            value = post.get(f"alt:{slot['key']}")
            if value in {course.code for course in slot["options"]}:
                picked[slot["key"]] = value
    return picked


def past_alternative_slots(student: Student) -> list[tuple[str, list[dict]]]:
    """"X or Y" chart slots in semesters already done, oldest first."""
    planning = (student.current_year, student.current_semester)
    slots = alternative_slots(student.programme)
    return sorted(((label, rows) for label, rows in slots.items() if semester_key(label) < planning), key=lambda item: semester_key(item[0]))


def mark_picked(semesters: list[tuple[str, list[dict]]], chosen: dict[str, str]) -> list[tuple[str, list[dict]]]:
    """Copy the student's existing pick onto each slot (templates can't look up a dict by a variable key)."""
    for _, slots in semesters:
        for slot in slots:
            slot["picked"] = chosen.get(slot["key"], "")
    return semesters


def completed_choices(student: Student | None, programme: Programme, planning: tuple[int, int],
                      admission_year: int) -> list[tuple[str, str]]:
    """(code, "CODE Title (1-2)") for the did well / struggled pickers: the student's completed courses, or for a new
    profile the chart's courses before the planned semester (the same ones saving will fill in), by code."""
    if student is not None:
        rows = StudentCourse.objects.filter(student=student, status="completed").select_related("course")
        found = {row.course.code: (row.course, row.semester_taken) for row in rows}
    else:  # why: the chart comes from the ID, so the first screen can ask too (typed-in electives come later)
        found = {}
        # same rule as rebuild_inferred_courses: no chart for the 2026+ curriculum
        for slot in course_slots(programme) if admission_year < 2026 else []:
            if (slot.year, slot.semester) < planning:
                found.setdefault(slot.course.code, (slot.course, f"{slot.year}-{slot.semester}"))
    return [(code, f"{code} {course_title(course.title)}" + (f" ({taken})" if taken else ""))
            for code, (course, taken) in sorted(found.items())]


def needs_electives_page(student: Student) -> bool:
    return bool(past_elective_semesters(student) or past_alternative_slots(student))


@login_required
def profile(request: HttpRequest) -> HttpResponse:
    student = Student.objects.filter(user=request.user).first()
    try:
        parsed = parse_bits_id(request.user.username)
    except BitsIdError:  # e.g. an admin account made with createsuperuser
        return HttpResponse("This account's username isn't a BITS ID, so it has no student profile.", status=400)
    programme = student.programme if student else resolve_programme(parsed)
    # an M.Sc.-only ID may still get (or already have) a B.E.: ask for it; its chart runs a 5th year
    second_degrees = second_degree_options(parsed.first_code) if not parsed.second_code else []
    planning = semester_for(parsed.admission_year)
    problem = semester_problem(planning, parsed.admission_year, 5 if second_degrees else max_year(programme))
    if problem:
        return render(request, "students/profile.html", {"problem": problem, "id_parts": id_parts(request.user.username, parsed), "programme": programme})
    label = f"{planning[0]}-{planning[1]}"
    form = ProfileForm(request.POST if request.method == "POST" else None, instance=student, planning=planning,
                       second_degrees=second_degrees, completed=completed_choices(student, programme, planning, parsed.admission_year))
    slots = alternative_slots(programme).get(label, [])
    if request.method == "POST" and form.is_valid():
        with transaction.atomic():
            student = form.save_for(request.user, parsed)
            # only the planned semester's "X or Y" picks are asked here; past ones are asked on the electives page
            student.alternative_choices.update(read_alternative_choices(request.POST, {label: slots}, [label]))
            student.save(update_fields=["alternative_choices"])
            refresh_courses(student)
        messages.success(request, "Profile saved.")
        return redirect("electives" if needs_electives_page(student) else "home")
    return render(request, "students/profile.html", {
        "form": form, "programme": programme, "id_parts": id_parts(request.user.username, parsed), "planning": label,
        "alternatives": mark_picked([(label, slots)] if slots else [], student.alternative_choices if student else {}),
    })


def semester_problem(planning: tuple[int, int] | None, admission_year: int, last_year: int) -> str:
    """Why this batch can't be planned with the loaded timetable, or "" if it can."""
    if planning is None:
        return "No timetable is loaded yet, so there's no semester to plan."
    if planning[0] < 1:
        return f"The {admission_year} batch hasn't started yet: the loaded timetable is for an earlier semester."
    if planning[0] > last_year:
        return f"By the loaded timetable, the {admission_year} batch has finished its {last_year}-year programme."
    return ""


def refresh_courses(student: Student) -> None:
    """After the planned semester or programme changes: drop electives that are no longer in the past, re-infer the
    chart's compulsory courses, and re-sort typed-in electives (a new second degree changes what they count as)."""
    StudentCourse.objects.filter(student=student, source="user", status="completed").exclude(
        semester_taken__in=past_elective_semesters(student)
    ).delete()
    rebuild_inferred_courses(student)
    categories = category_map(student.programme)
    typed = list(StudentCourse.objects.filter(student=student, source="user").select_related("course"))
    for row in typed:
        row.category = categories.get(row.course.code, row.category)
    StudentCourse.objects.bulk_update(typed, ["category"])


def sync_semester(student: Student) -> None:
    """Move the student to the semester the loaded timetable implies (e.g. after a new timetable is ingested)."""
    planning = semester_for(student.admission_year)
    if planning is None or planning == (student.current_year, student.current_semester) or semester_problem(
        planning, student.admission_year, max_year(student.programme)
    ):
        return
    with transaction.atomic():
        # courses added from the assistant were this semester's; once it's over they count as done
        StudentCourse.objects.filter(student=student, status="current", source="user").update(status="completed")
        student.current_year, student.current_semester = planning
        student.minor_registered = student.minor is not None and planning[0] >= MINOR_FROM_YEAR
        student.save(update_fields=["current_year", "current_semester", "minor_registered"])
        refresh_courses(student)


@login_required
def electives(request: HttpRequest) -> HttpResponse:
    """One formset per past semester (2-1 onwards). Saving replaces that student's typed-in electives."""
    student = current_student(request)
    if student is None:
        return redirect("profile")
    past_alternatives = past_alternative_slots(student)
    if request.method == "POST" and past_alternatives:
        # why first: a changed pick changes which chart courses are done, which the checks below rely on
        picked = read_alternative_choices(request.POST, dict(past_alternatives), [label for label, _ in past_alternatives])
        if picked != {key: student.alternative_choices.get(key) for key in picked}:
            student.alternative_choices.update(picked)
            student.save(update_fields=["alternative_choices"])
            rebuild_inferred_courses(student)
    choices = elective_choices(student.programme, student.admission_year)
    codes_by_category = {category: {course.code for course in courses} for category, courses in choices.items()}
    saved = StudentCourse.objects.filter(student=student, source="user", status="completed").select_related("course")
    from_chart = dict(StudentCourse.objects.filter(student=student, source="pattern").values_list("course__code", "semester_taken"))

    formsets = []
    for semester in past_elective_semesters(student):
        initial = [
            {"category": _category_of(row.course.code, codes_by_category), "course": row.course.code}
            for row in saved if row.semester_taken == semester
        ]
        formset = ElectiveFormSet(
            request.POST if request.method == "POST" else None, initial=initial, prefix=f"s{semester.replace('-', '')}",
            form_kwargs={"codes_by_category": codes_by_category},
        )
        formsets.append((semester, formset))

    if request.method == "POST" and all(formset.is_valid() for _, formset in formsets):
        rows = [(semester, form.cleaned_data) for semester, formset in formsets for form in formset if form.cleaned_data.get("course")]
        codes = [data["course"] for _, data in rows]
        duplicates = sorted({code for code in codes if codes.count(code) > 1})
        already_done = sorted(f"{code} ({from_chart[code]})" for code in set(codes) & from_chart.keys())
        error = (f"You entered {', '.join(duplicates)} more than once." if duplicates else
                 f"Already counted from your programme chart: {', '.join(already_done)}." if already_done else "")
        if error:
            return render(request, "students/electives.html", {
                "formsets": formsets, "choices": choices, "error": error, "needs": elective_needs(student, codes),
                "alternatives": mark_picked(past_alternatives, student.alternative_choices),
            })
        courses = {course.code: course for course in Course.objects.filter(code__in=codes)}
        with transaction.atomic():
            saved.delete()
            StudentCourse.objects.bulk_create(
                StudentCourse(
                    student=student, course=courses[data["course"]], status="completed", semester_taken=semester,
                    source="user",
                    category=_category_of(data["course"], codes_by_category),
                )
                for semester, data in rows
            )
        messages.success(request, "Courses saved.")
        return redirect("home")
    return render(request, "students/electives.html", {
        "formsets": formsets, "choices": choices, "alternatives": mark_picked(past_alternatives, student.alternative_choices),
        "needs": elective_needs(student),
    })


def _category_of(code: str, codes_by_category: dict[str, set[str]]) -> str:
    return next((category for category, codes in codes_by_category.items() if code in codes), "")


def applicable_gaps(student: Student) -> list[str]:
    """User messages of the KnownGaps specific to this student (by programme, campus or batch).

    why no programmes == "all" gaps: a warning every student sees tells nobody anything; those gaps
    (e.g. unstated batch range) are surfaced where they matter instead (the needs_review first-year courses).
    """
    names = {student.programme.name}
    for part in (student.programme.first_component, student.programme.second_component):
        if part:
            names.add(part.name)
    messages = []
    for gap in KnownGap.objects.exclude(user_message__isnull=True).exclude(user_message=""):
        affected = gap.affected
        programmes = affected.get("programmes")
        batches = " ".join(affected.get("batches", []))
        if (
            (isinstance(programmes, list) and names & set(programmes))
            or student.campus in affected.get("campuses", [])
            # ponytail: only understands "<year> onwards"; add parsing when a gap lists other batch shapes
            or (batches.endswith("onwards") and student.admission_year >= int(batches.split()[0]))
        ):
            messages.append(gap.user_message)
    return messages


CATEGORY_ORDER = ("CDC", "GIR", "DEL", "HUEL", "OPEL", "AUDIT")
# why: CHART courses (named in the chart but in no Bulletin list) are compulsory for the degree, so they read as
# CDCs; recommender/ keeps them apart for the requirement counts
DISPLAY_CATEGORY = {"CHART": "CDC"}
# (short form, full form, meaning). "GIR" (General Institutional Requirement) is shown as "Foundation": students
# didn't recognise the acronym (2026-09-26)
CATEGORIES = {
    "CDC": ("CDC", "Compulsory Disciplinary Course", "Core courses your degree requires."),
    "GIR": ("Foundation", "Institute foundation course", "Taken by every BITS student, whatever the branch: the science, maths, engineering and technical-arts basics, plus courses like Environmental Studies. Officially the General Institutional Requirement (GIR). HUELs officially belong to it too, but you choose those, so they're listed on their own."),
    "DEL": ("DEL", "Disciplinary Elective", "Electives from your discipline's list."),
    "HUEL": ("HUEL", "Humanities Elective", "Electives from the humanities pool."),
    "OPEL": ("OPEL", "Open Elective", "Any other course; extra DELs and HUELs count here too."),
    "AUDIT": ("Audit", "Audit course", "No credit; never counted."),
}
KEY = [(code, *CATEGORIES[code]) for code in CATEGORY_ORDER]


def display_category(category: str) -> str:
    return DISPLAY_CATEGORY.get(category, category)


def semester_key(label: str) -> tuple[int, int]:
    year, sem = label.split("-")
    return int(year), int(sem)


def attach_display(rows: list[StudentCourse]) -> None:
    """Give each row its display category and, for "X or Y" chart slots, the alternative Course objects."""
    alt_codes = {code for row in rows if row.pattern_slot for code in row.pattern_slot.alternatives}
    alt_courses = {course.code: course for course in Course.objects.filter(code__in=alt_codes)}
    for row in rows:
        row.cat = display_category(row.category)
        row.tag = CATEGORIES.get(row.cat, (row.cat,))[0]
        codes = row.pattern_slot.alternatives if row.pattern_slot else []
        row.alternatives = [alt_courses.get(code) or Course(code=code) for code in codes]
        # "X or Y" slot: both options as halves; `chosen` marks the student's pick (empty = not answered yet)
        if codes:
            row.options = [row.pattern_slot.course] + row.alternatives
            row.chosen = row.course.code if row.pattern_slot.course.code in row.student.alternative_choices else ""


def planned_rows(student: Student) -> list[StudentCourse]:
    """Unsaved rows for the chart's courses after the planned semester, so the chart can show what's ahead."""
    planning = (student.current_year, student.current_semester)
    categories = category_map(student.programme)
    chosen = {course.code: course for course in Course.objects.filter(code__in=student.alternative_choices.values())}
    rows: dict[str, StudentCourse] = {}
    for slot in course_slots(student.programme):
        if (slot.year, slot.semester) <= planning:
            continue
        course = chosen.get(student.alternative_choices.get(slot.course.code), slot.course)
        rows.setdefault(course.code, StudentCourse(
            student=student, course=course, pattern_slot=slot, status="planned",
            semester_taken=f"{slot.year}-{slot.semester}", category=categories.get(course.code, ""),
        ))
    return list(rows.values())


def elective_slots(student: Student) -> dict[tuple[int, int], list[dict]]:
    """The chart's elective placeholders from the planned semester on: {(3, 1): [{"cat": "DEL", "label": ...}]}."""
    planning = (student.current_year, student.current_semester)
    slots: dict[tuple[int, int], list[dict]] = {}
    for slot in chart_slots(student.programme, "elective"):
        if (slot.year, slot.semester) < planning:
            continue
        text = slot.elective_category
        cat = "DEL" if text.startswith("DEL") else text if text in ("HUEL", "OPEL") else "OPEL"
        label = "OPEL / HUEL" if text == "Open/Humanities" else text
        slots.setdefault((slot.year, slot.semester), []).append({"cat": cat, "label": label})
    return slots


def chart_grid(rows: list[StudentCourse], student: Student) -> list[dict]:
    """Rows laid out like the Bulletin's chart: one entry per year of the programme, each with its two semesters."""
    by_semester: dict[tuple[int, int], list[StudentCourse]] = {}
    for row in rows:
        by_semester.setdefault(semester_key(row.semester_taken), []).append(row)
    electives = elective_slots(student)
    planning = (student.current_year, student.current_semester)
    return [
        {"year": year, "semesters": [
            {"label": f"{year}-{sem}", "rows": by_semester.get((year, sem), []), "electives": electives.get((year, sem), []),
             "is_current": (year, sem) == planning, "is_future": (year, sem) > planning}
            for sem in (1, 2)
        ]}
        for year in range(1, max_year(student.programme) + 1)
    ]


def sort_rows(rows: list[StudentCourse]) -> None:
    def order(row: StudentCourse) -> tuple:
        cat = display_category(row.category)
        return semester_key(row.semester_taken), CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else 99, row.course.code
    rows.sort(key=order)


def current_student(request: HttpRequest) -> Student | None:
    student = Student.objects.filter(user=request.user).select_related("programme", "minor").first()
    if student:
        sync_semester(student)
    return student


@login_required
def home(request: HttpRequest) -> HttpResponse:
    """Profile summary + every done / current / upcoming course, laid out as the programme chart or grouped by kind."""
    student = current_student(request)
    if student is None:
        return redirect("profile")
    rows = list(StudentCourse.objects.filter(student=student).select_related("course", "pattern_slot__course", "student"))
    rows += planned_rows(student) if rows else []
    sort_rows(rows)
    attach_display(rows)

    done_rows = [row for row in rows if row.status == "completed"]
    done_units = sum(row.course.units or 0 for row in done_rows)
    groups = []
    for code in CATEGORY_ORDER:
        group_rows = [row for row in rows if row.cat == code and row.status != "planned"]
        if group_rows:
            short, full, meaning = CATEGORIES[code]
            units = sum(row.course.units or 0 for row in group_rows if row.status == "completed")
            groups.append({
                "code": code, "short": short, "full": full, "meaning": meaning, "rows": group_rows, "done_units": units,
                "done_count": sum(row.status == "completed" for row in group_rows),
                "share": round(100 * units / done_units, 2) if done_units else 0,
            })
    current = [row.course.code for row in rows if row.status == "current"]
    week = timetable_context(current, categories=category_map(student.programme)) if current else {}
    return render(request, "students/home.html", {
        "student": student, "warnings": applicable_gaps(student),
        "week_grid": (week.get("timetables") or [None])[0], "week": week.get("week"), "today": timezone.localdate(),
        "planning": f"{student.current_year}-{student.current_semester}",
        "grid": chart_grid(rows, student) if rows else [], "groups": groups, "key": KEY,
        "done_count": len(done_rows), "done_units": done_units,
        "has_elective_semesters": needs_electives_page(student),
        "needs": elective_needs(student), "minor": minor_progress(student), "compulsory_left": compulsory_left(student),
        "load": semester_load(student),
        "is_2026_batch": student.admission_year >= 2026,
    })


@login_required
@require_POST
def remaining_preview(request: HttpRequest) -> HttpResponse:
    """The requirement counts for the electives typed on the electives page, before they're saved."""
    student = current_student(request)
    if student is None:
        return HttpResponse(status=400)
    typed = [" ".join(value.upper().split()) for name, value in request.POST.items() if name.endswith("-course") and value.strip()]
    return render(request, "students/_needs.html", {"needs": elective_needs(student, typed)})


@login_required
def semester(request: HttpRequest) -> HttpResponse:
    """This semester's courses by kind, the best clash-free timetables for them, and their exam calendar."""
    student = current_student(request)
    if student is None:
        return redirect("profile")
    rows = list(StudentCourse.objects.filter(student=student, status="current").select_related("course", "pattern_slot__course", "student"))
    sort_rows(rows)
    attach_display(rows)
    codes = [row.course.code for row in rows]
    grids = timetable_context(codes, filters=read_filters(request.GET), categories=category_map(student.programme))
    offered = grids.pop("offered")
    return render(request, "students/semester.html", {
        "student": student, "planning": f"{student.current_year}-{student.current_semester}", "key": KEY,
        "rows": rows, "electives": elective_strip(student, rows), "load": semester_load(student),
        "not_offered": [row.course for row in rows if row.course.code not in {offering.course.code for offering in offered}],
        **grids,
        "exams": [exam_calendar(offered, "Midsem"), exam_calendar(offered, "Compre")] if offered else [],
    })


def elective_strip(student: Student, rows: list[StudentCourse]) -> list[dict]:
    """One chip per elective kind: how many are still needed and how many were added this semester.
    Kinds the degree doesn't require (a dual degree's OPEL) are left out."""
    status = requirement_status(student)
    strip = []
    for code in ("DEL", "HUEL", "OPEL"):
        need = status[code]
        if not need.required_courses:
            continue
        strip.append({"code": code, "short": CATEGORIES[code][0], "full": CATEGORIES[code][1],
                      "added": sum(row.cat == code for row in rows), "left": need.remaining_courses,
                      "complete": need.complete, "query": f"Suggest a {code} for this semester"})
    return strip


def read_filters(query) -> TimetableFilters:
    """The filter form above the timetables (GET params); anything malformed is ignored."""
    teachers = [tuple(value.split("|", 1)) for value in query.getlist("teacher") if "|" in value]
    avoid = set()
    for value in query.getlist("avoid"):
        day, _, period = value.rpartition("-")
        if day in DAYS and period.isdigit():
            avoid.add((day, int(period)))
    return TimetableFilters(no_8am=query.get("no_8am") == "1", compact=query.get("compact") == "1",
                            free_day=query.get("free_day") if query.get("free_day") in DAYS else "",
                            teachers=teachers, avoid=avoid)


def filter_notes(filters: TimetableFilters, best) -> list[str]:
    """Say when the best timetable still can't meet a filter (filters sort, they never hide timetables)."""
    notes = []
    missing = [f"{name} for {code}" for code, name in filters.teachers if not best.teaches(code, name)]
    if missing:
        notes.append(f"No timetable fits with {', '.join(missing)}; these are the closest.")
    blocked = sum(slot in filters.avoid for slot in best.by_slot)
    if blocked:
        notes.append(f"No timetable keeps all the times you picked free; the best ones use {blocked} of them.")
    eight_ams = best.score()[1]
    if filters.no_8am and eight_ams:
        notes.append(f"No timetable avoids 8 AM completely; the best ones have {eight_ams} class{'es' if eight_ams > 1 else ''} at 8 AM.")
    on_day = sum(day == filters.free_day for day, _ in best.by_slot)
    if filters.free_day and on_day:
        notes.append(f"No timetable keeps {DAY_NAMES[filters.free_day]} free; the best ones have {on_day} "
                     f"class hour{'s' if on_day > 1 else ''} that day.")
    return notes


def teacher_options(offered: list) -> list[dict]:
    """[{code, names}] of every course with more than one teacher to choose from."""
    options = []
    for offering in offered:
        names = sorted({name for section in offering.sections.all() if not section.cancelled for name in section.instructors})
        if len(names) > 1:
            options.append({"code": offering.course.code, "names": names})
    return options


def timetable_context(codes: list[str], highlight: set[str] | None = None, filters: TimetableFilters | None = None,
                      categories: dict[str, str] | None = None) -> dict:
    """Everything students/_timetables.html needs for these courses: the best clash-free week grids for the
    filters (or the pairs that always clash), plus `offered` (their Offerings, for the exam calendar).
    categories (code -> category) colours the grid by course category."""
    filters = filters or TimetableFilters()
    offerings = offerings_for(codes)
    offered = [offerings[code] for code in codes if code in offerings]
    timetables = generate(offered, order=filters.key)
    filters.prefer_teachers(list({id(pick): pick for tt in timetables for pick in tt.picks}.values()))
    last_period = max([10] + [period for tt in timetables for _, period in tt.by_slot])
    times = period_times()
    for pick in {id(pick): pick for tt in timetables for pick in tt.picks}.values():
        pick.when = timing_text(pick.section.timings, times)
        pick.cat = display_category((categories or {}).get(pick.code, ""))
        pick.letter = pick.type[0].upper()  # L / T / P
    return {
        "offered": offered, "highlight": highlight or set(), "filters": filters,
        "filter_notes": filter_notes(filters, timetables[0]) if timetables else [],
        "teacher_options": teacher_options(offered),
        "chosen_teachers": [(f"{code}|{name}", code, name) for code, name in filters.teachers],
        "avoid_keys": {f"{day}-{period}" for day, period in filters.avoid},
        "periods": [(str(period), times[period]) for period in range(1, last_period + 1)],
        "day_names": [(day, DAY_NAMES[day]) for day in DAYS],
        "timetables": [{"rows": week_rows(tt, last_period), "picks": sorted(tt.picks, key=lambda pick: (pick.code, pick.type)),
                        "score": dict(zip(("gaps", "eight_ams", "days"), tt.score())),
                        "has_alternatives": any(pick.others for pick in tt.picks)} for tt in timetables],
        "clashes": always_clashing(offered) if offered and not timetables else [],
        "week": list(zip(DAYS, week_dates(timezone.localdate()))), "today": timezone.localdate(),
    }


def handout_extra(handout_text: str, bulletin_text: str) -> str:
    """The handout's description minus what repeats the Bulletin's (already shown under "About").
    Handouts usually open with the Bulletin text, often with spelling changes ("Behavioural"), so the repeat is
    found by word overlap, not exact match."""
    text = " ".join(handout_text.split())
    words = lambda value: re.findall(r"[a-z]+", value.lower())  # noqa: E731
    bulletin = words(bulletin_text)
    if not bulletin:
        return text
    head = words(text)[: len(bulletin)]
    if SequenceMatcher(None, head, bulletin).ratio() < 0.85:
        return text
    # drop the sentences that make up the repeated part; keep whatever the handout adds after it
    sentences = re.split(r"(?<=[.])\s+", text)
    kept, covered = [], 0
    for sentence in sentences:
        if covered < len(bulletin):
            covered += len(words(sentence))
        else:
            kept.append(sentence)
    return " ".join(kept)


@login_required
def course(request: HttpRequest, code: str) -> HttpResponse:
    """Everything we hold on one course: Bulletin data, this semester's sections and exams, and its handouts."""
    student = current_student(request)
    if student is None:
        return redirect("profile")
    course = get_object_or_404(Course, code=code)
    cat = display_category(category_map(student.programme).get(course.code, ""))
    offering = offerings_for([course.code]).get(course.code)
    times = period_times()
    sections: dict[str, list[dict]] = {}
    if offering:
        for section in offering.sections.all().order_by("pk"):
            sections.setdefault(section.type, []).append({"section": section, "when": timing_text(section.timings, times)})
    prereq_codes = {code for group in course.prerequisites or [] for code in group}
    known = set(Course.objects.filter(code__in=prereq_codes).values_list("code", flat=True))
    taken = StudentCourse.objects.filter(student=student, course=course).first()
    handouts = list(course.handouts.all().order_by("file"))
    for handout in handouts:
        handout.extra = handout_extra(handout.description, course.description)
    return render(request, "students/course.html", {
        "course": course, "cat": cat, "category": CATEGORIES.get(cat), "key": KEY,
        "offering": offering, "sections": sections,
        "midsem": exam_date(offering.midsem_date, offering.semester_tag) if offering else None,
        "compre": exam_date(offering.compre_date, offering.semester_tag) if offering else None,
        "prerequisites": [[(code, code in known) for code in group] for group in course.prerequisites or []],
        "handouts": handouts, "taken": taken,
    })


@login_required
@require_POST
def remove_course(request: HttpRequest) -> HttpResponse:
    """Take out a course added from the course assistant (this semester's, typed-in rows only)."""
    student = current_student(request)
    if student is None:
        return redirect("profile")
    deleted, _ = StudentCourse.objects.filter(student=student, status="current", source="user",
                                              course__code=request.POST.get("code", "")).delete()
    if deleted:
        messages.success(request, f"Removed {request.POST['code']}.")
    return redirect("semester")
