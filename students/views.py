"""Register -> profile -> past electives -> home (profile summary). Recommendations come later."""
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import Max
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render

from catalog.models import Course, KnownGap, PatternSlot, Programme
from recommender.categories import elective_choices
from recommender.history import alternative_slots, rebuild_inferred_courses
from recommender.requirements import compulsory_left, elective_needs, minor_progress, semester_load

from .bits_id import BRANCH_CODES, BitsIdError, ParsedId, parse_bits_id, resolve_programme
from .forms import ElectiveFormSet, ProfileForm, RegisterForm
from .models import Student, StudentCourse

FIRST_ELECTIVE_SEMESTER = (2, 1)  # electives start in 2-1; earlier semesters are all compulsory


def register(request: HttpRequest) -> HttpResponse:
    form = RegisterForm(request.POST or None)
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
    form = ProfileForm(request.POST or None, instance=student, max_year=max_year(programme))
    slots_by_semester = alternative_slots(programme)
    if request.method == "POST" and form.is_valid():
        with transaction.atomic():
            student = form.save_for(request.user, parsed)
            # only the planned semester's "X or Y" picks are asked here; past ones are asked on the electives page
            student.alternative_choices.update(read_alternative_choices(request.POST, slots_by_semester, [form.cleaned_data["planning"]]))
            student.save(update_fields=["alternative_choices"])
            # why: electives dated at or after the planned semester can't be "done" any more
            StudentCourse.objects.filter(student=student, source="user", status="completed").exclude(
                semester_taken__in=past_elective_semesters(student)
            ).delete()
            rebuild_inferred_courses(student)
        messages.success(request, "Profile saved.")
        return redirect("electives" if needs_electives_page(student) else "home")
    return render(request, "students/profile.html", {
        "form": form, "programme": programme, "id_parts": id_parts(request.user.username, parsed),
        "alternatives": mark_picked(sorted(slots_by_semester.items(), key=lambda item: semester_key(item[0])),
                                    student.alternative_choices if student else {}),
    })


@login_required
def electives(request: HttpRequest) -> HttpResponse:
    """One formset per past semester (2-1 onwards). Saving replaces that student's typed-in electives."""
    student = Student.objects.filter(user=request.user).first()
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
            {"category": _category_of(row.course.code, codes_by_category), "course": row.course.code,
             "grade": row.grade or "", "comfort": row.comfort}
            for row in saved if row.semester_taken == semester
        ]
        formset = ElectiveFormSet(
            request.POST or None, initial=initial, prefix=f"s{semester.replace('-', '')}",
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
                "formsets": formsets, "choices": choices, "error": error,
                "alternatives": mark_picked(past_alternatives, student.alternative_choices),
            })
        courses = {course.code: course for course in Course.objects.filter(code__in=codes)}
        with transaction.atomic():
            saved.delete()
            StudentCourse.objects.bulk_create(
                StudentCourse(
                    student=student, course=courses[data["course"]], status="completed", semester_taken=semester,
                    source="user", grade=data["grade"] or None, comfort=data["comfort"],
                    category=_category_of(data["course"], codes_by_category),
                )
                for semester, data in rows
            )
        messages.success(request, "Courses saved.")
        return redirect("home")
    return render(request, "students/electives.html", {
        "formsets": formsets, "choices": choices, "alternatives": mark_picked(past_alternatives, student.alternative_choices),
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


CATEGORY_ORDER = ("CDC", "GIR", "CHART", "DEL", "HUEL", "OPEL", "AUDIT")
# (name, short tag on a course tile, one-line meaning). GIR is BITS's "General Institutional Requirement".
CATEGORIES = {
    "CDC": ("Discipline core", "CDC", "Compulsory courses of your degree (CDCs)."),
    "GIR": ("Common core", "Common", "Courses every BITS student takes, whatever the branch (the Bulletin's General Institutional Requirements)."),
    "CHART": ("Other compulsory", "Compulsory", "Named in your programme's chart but in no Bulletin list, e.g. the first-year maths courses."),
    "DEL": ("Discipline electives", "DEL", "Electives from your discipline's list."),
    "HUEL": ("Humanities electives", "HUEL", "Electives from the humanities pool."),
    "OPEL": ("Open electives", "OPEL", "Any other course."),
    "AUDIT": ("Audit", "Audit", "No credit; never counted."),
}


def semester_key(label: str) -> tuple[int, int]:
    year, sem = label.split("-")
    return int(year), int(sem)


def attach_display(rows: list[StudentCourse]) -> None:
    """Give each row its tile tag and, for "X or Y" chart slots, the alternative Course objects."""
    alt_codes = {code for row in rows if row.pattern_slot for code in row.pattern_slot.alternatives}
    alt_courses = {course.code: course for course in Course.objects.filter(code__in=alt_codes)}
    for row in rows:
        row.tag = CATEGORIES.get(row.category, ("", row.category, ""))[1]
        codes = row.pattern_slot.alternatives if row.pattern_slot else []
        row.alternatives = [alt_courses.get(code) or Course(code=code) for code in codes]
        # "X or Y" slot: both options as halves; `chosen` marks the student's pick (empty = not answered yet)
        if codes:
            row.options = [row.pattern_slot.course] + row.alternatives
            row.chosen = row.course.code if row.pattern_slot.course.code in row.student.alternative_choices else ""


def chart_grid(rows: list[StudentCourse], student: Student) -> list[dict]:
    """Rows laid out like the Bulletin's chart: one entry per year, each with its two semesters."""
    by_semester: dict[tuple[int, int], list[StudentCourse]] = {}
    for row in rows:
        by_semester.setdefault(semester_key(row.semester_taken), []).append(row)
    planning = (student.current_year, student.current_semester)
    return [
        {"year": year, "semesters": [
            {"label": f"{year}-{sem}", "rows": by_semester.get((year, sem), []), "is_current": (year, sem) == planning,
             "is_future": (year, sem) > planning}
            for sem in (1, 2)
        ]}
        for year in range(1, student.current_year + 1)
    ]


@login_required
def home(request: HttpRequest) -> HttpResponse:
    """Profile summary + every done / current course, laid out as the programme chart or grouped by kind."""
    student = Student.objects.filter(user=request.user).select_related("programme", "minor").first()
    if student is None:
        return redirect("profile")
    rows = list(StudentCourse.objects.filter(student=student).select_related("course", "pattern_slot__course", "student"))
    rows.sort(key=lambda row: (semester_key(row.semester_taken), CATEGORY_ORDER.index(row.category) if row.category in CATEGORY_ORDER else 99, row.course.code))
    attach_display(rows)

    done_rows = [row for row in rows if row.status == "completed"]
    done_units = sum(row.course.units or 0 for row in done_rows)
    groups = []
    for code in CATEGORY_ORDER:
        group_rows = [row for row in rows if row.category == code]
        if group_rows:
            name, tag, meaning = CATEGORIES[code]
            units = sum(row.course.units or 0 for row in group_rows if row.status == "completed")
            groups.append({
                "code": code, "name": name, "meaning": meaning, "rows": group_rows, "done_units": units,
                "done_count": sum(row.status == "completed" for row in group_rows),
                "share": round(100 * units / done_units, 2) if done_units else 0,
            })
    return render(request, "students/home.html", {
        "student": student, "warnings": applicable_gaps(student),
        "planning": f"{student.current_year}-{student.current_semester}",
        "grid": chart_grid(rows, student) if rows else [], "groups": groups,
        "done_count": len(done_rows), "done_units": done_units,
        "has_elective_semesters": needs_electives_page(student),
        "needs": elective_needs(student), "minor": minor_progress(student), "compulsory_left": compulsory_left(student),
        "load": semester_load(student),
        "is_2026_batch": student.admission_year >= 2026,
    })
