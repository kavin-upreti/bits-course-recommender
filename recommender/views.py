"""The chat page and its endpoint (todo.md section 11). No history: each message is answered on its own."""
import json
import logging
import threading

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.urls import reverse
from django.views.decorators.http import require_GET, require_POST

from catalog.models import Course, Offering
from students.models import Student, StudentCourse
from students.templatetags.text import course_title
from students.views import read_filters, timetable_context

from . import config
from .agent import run_agent
from .categories import category_map
from .context import build_context
from .codes import normalise_code, resolve_course
from .embeddings import embed_query
from .history import done_codes
from .plan import check_plan, offered
from .tools import DISPLAY_CATEGORY

logger = logging.getLogger(__name__)
_warmed = threading.Event()
SELECTION = "plan_selection"  # session key: codes ticked on the chat page, kept until finalised or logout

EXAMPLES = [  # from the brief
    "Suggest DELs related to AI",
    "I want an OPEL with no attendance requirement",
    "Suggest courses with no midsem and a lenient makeup policy",
    "I need a HUEL and prefer project-based evaluation",
]


@login_required
def chat(request: HttpRequest) -> HttpResponse:
    if not Student.objects.filter(user=request.user).exists():
        return redirect("profile")
    student = Student.objects.get(user=request.user)
    if not _warmed.is_set():
        _warmed.set()
        # why: loading the embedding model takes seconds; do it while the student is still typing
        threading.Thread(target=warm_up, daemon=True).start()
    return render(request, "recommender/chat.html", {"examples": EXAMPLES, "debug": settings.DEBUG,
                                                     "selection": selection_items(student, request.session.get(SELECTION, []))})


def warm_up() -> None:
    """Load the embedding model once per process (never raises). No database work here: a second thread's
    SQLite read can collide with the request's writes."""
    try:
        embed_query("warm up")
    except Exception:  # why broad: a failed warm-up only means the first question loads it instead
        logger.exception("Embedding warm-up failed")


@login_required
@require_POST
def recommend(request: HttpRequest) -> JsonResponse:
    """POST {"message": "..."} -> {"reply", "cards"} (+ "debug" when settings.DEBUG)."""
    student = Student.objects.filter(user=request.user).select_related("programme", "minor").first()
    if student is None:
        return JsonResponse({"error": "Please create your profile first."}, status=400)
    try:
        message = json.loads(request.body).get("message")
    except (ValueError, AttributeError):
        return JsonResponse({"error": "Send JSON like {\"message\": \"...\"}."}, status=400)
    if not isinstance(message, str):
        return JsonResponse({"error": "The message is missing."}, status=400)
    result = run_agent(student, message)
    body = {"reply": result.reply, "cards": result.cards}
    if settings.DEBUG:
        body["debug"] = result.debug
    return JsonResponse(body)


# ---------------------------------------------------------------- picking courses (select -> preview -> finalise)

def body_json(request: HttpRequest) -> dict:
    try:
        data = json.loads(request.body)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def selection_items(student: Student, codes: list[str]) -> list[dict]:
    """[{code, title, category}] for the right-hand list and the finalise popup."""
    categories = category_map(student.programme)
    courses = {course.code: course for course in Course.objects.filter(code__in=codes)}
    return [{"code": code, "title": course_title(courses[code].title),
             "category": DISPLAY_CATEGORY.get(categories.get(code, ""), categories.get(code, ""))}
            for code in codes if code in courses]


def pickable(student: Student, raw_code: str) -> tuple[Course | None, str]:
    """The course if it can be added this semester, else (None, why not)."""
    course = resolve_course(normalise_code(raw_code or ""))
    if course is None:
        return None, "No such course."
    tag = Offering.objects.order_by("pk").values_list("semester_tag", flat=True).first() or ""
    if not offered(course, tag):
        return None, f"{course.code} isn't offered this semester."
    if course.code in done_codes(student):
        return None, f"{course.code} (or an equivalent) is already done or in this semester."
    return course, ""


def plan_problem(student: Student, codes: list[str]) -> str:
    """Why these courses can't all be taken with the current ones ("" = some timetable works). Tries every section
    combination and checks exams, units and the higher-degree limit too (the same check the recommender uses)."""
    plan = check_plan(build_context(student), codes)
    return "" if plan["ok"] or plan.get("limit_hit") else plan["problem"]


def logged_in_student(request: HttpRequest) -> Student | None:
    return Student.objects.filter(user=request.user).select_related("programme").first()


@login_required
@require_POST
def select_course(request: HttpRequest) -> JsonResponse:
    """POST {"code", "selected": bool} -> the updated selection."""
    student = logged_in_student(request)
    if student is None:
        return JsonResponse({"error": "Please create your profile first."}, status=400)
    data = body_json(request)
    codes = list(request.session.get(SELECTION, []))
    if data.get("selected"):
        course, problem = pickable(student, data.get("code"))
        if course is None:
            return JsonResponse({"error": problem}, status=400)
        if course.code not in codes:
            if len(codes) >= config.MAX_PLAN_COURSES:
                return JsonResponse({"error": f"You can select at most {config.MAX_PLAN_COURSES} courses."}, status=400)
            if problem := plan_problem(student, codes + [course.code]):
                return JsonResponse({"error": f"Can't add {course.code}: {problem}"}, status=400)
            codes.append(course.code)
    else:
        codes = [code for code in codes if code != normalise_code(data.get("code") or "")]
    request.session[SELECTION] = codes
    return JsonResponse({"selection": selection_items(student, codes)})


@login_required
@require_GET
def plan_timetables(request: HttpRequest) -> HttpResponse:
    """HTML: every clash-free timetable of this semester's courses + the selected ones + ?with=CODE (dashed)."""
    student = logged_in_student(request)
    if student is None:
        return HttpResponse("Please create your profile first.", status=400)
    extra = [normalise_code(request.GET["with"])] if request.GET.get("with") else []
    considering = list(dict.fromkeys(request.session.get(SELECTION, []) + extra))
    current = list(StudentCourse.objects.filter(student=student, status="current").values_list("course__code", flat=True))
    codes = list(dict.fromkeys(current + considering))
    grids = timetable_context(codes, set(considering) - set(current), read_filters(request.GET), category_map(student.programme))
    grids.pop("offered")
    heading = f"Timetables with {', '.join(considering)}" if considering else "Your current timetables"
    # why: the grids only show class times; an exam clash or the unit limit blocks a course with a clash-free week too
    blocked = plan_problem(student, considering) if considering and grids["timetables"] else ""
    return render(request, "students/_timetables.html", {**grids, "heading": heading, "with_code": extra[0] if extra else "",
                                                         "filters_url": reverse("plan_timetables"), "blocked": blocked})


@login_required
@require_POST
def finalise(request: HttpRequest) -> JsonResponse:
    """POST {"codes": [...]}: add them to this semester's courses (removable on the semester page)."""
    student = logged_in_student(request)
    if student is None:
        return JsonResponse({"error": "Please create your profile first."}, status=400)
    codes = body_json(request).get("codes")
    if not isinstance(codes, list) or not codes:
        return JsonResponse({"error": "Nothing selected."}, status=400)
    categories = category_map(student.programme)
    courses = []
    for code in codes:
        course, problem = pickable(student, str(code))
        if course is None:
            return JsonResponse({"error": problem}, status=400)
        courses.append(course)
    # why again: the selection lives in the session and may predate a change (another tab, a finalise elsewhere)
    if problem := plan_problem(student, [course.code for course in courses]):
        return JsonResponse({"error": f"These can't all be taken together: {problem}"}, status=400)
    with transaction.atomic():
        for course in courses:
            StudentCourse.objects.get_or_create(student=student, course=course, defaults={
                "status": "current", "source": "user", "category": categories.get(course.code, ""),
                "semester_taken": f"{student.current_year}-{student.current_semester}"})
    added = {course.code for course in courses}
    request.session[SELECTION] = [code for code in request.session.get(SELECTION, []) if code not in added]
    return JsonResponse({"redirect": reverse("semester")})
