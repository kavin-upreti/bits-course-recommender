"""Per-student data (ideation Part 5, 'Student data'). Not from PDFs; never touched by `manage.py ingest`.

why PROTECT on catalog FKs: ingest clears and reloads the catalog; PROTECT makes it refuse instead of silently
deleting students' courses and plans (see the ingest command's --wipe-students).
"""
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models

from catalog.models import Course, Minor, PatternSlot, Programme, Section
from recommender.config import DAY_CODES, DAY_NAMES

GOALS = [("job", "Job / placement"), ("research", "Research / higher studies"), ("not_sure", "Not sure")]
GRADES = [(g, g) for g in ("A", "A-", "B", "B-", "C", "C-", "D", "E", "NC")]
COMFORT = [
    ("5", "Very comfortable"), ("4", "Comfortable"), ("3", "Moderate"),
    ("2", "Not too comfortable"), ("1", "Not good at all"),
]
# evaluation styles a student would rather avoid (profile checkboxes); ranked lower, never removed
EVAL_STYLES = [
    ("many_quizzes", "Lots of quizzes"),
    ("closed_book", "Only closed-book exams"),
    ("strict_attendance", "Strict attendance requirement"),
    ("heavy_compre", "Compre worth a lot"),
    ("no_makeup", "No makeup exams"),
]


def validate_eval_styles(value: list) -> None:
    """Only the known style keys, each at most once."""
    allowed = {key for key, _ in EVAL_STYLES}
    if not isinstance(value, list) or any(item not in allowed for item in value):
        raise ValidationError(f"Allowed evaluation styles: {', '.join(sorted(allowed))}.")
    if len(set(value)) != len(value):
        raise ValidationError("Each evaluation style can appear only once.")


class Student(models.Model):
    """Name = user.first_name, BITS ID = user.username. Campus, batch and programme come from the ID
    (students/bits_id.py); a dual ID maps to the combined dual Programme, which links both degrees."""

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="student")
    campus = models.CharField(max_length=30, default="Pilani")
    admission_year = models.PositiveSmallIntegerField()
    programme = models.ForeignKey(Programme, on_delete=models.PROTECT, related_name="+")
    current_year = models.PositiveSmallIntegerField()  # the semester being planned (ideation 6.1)
    current_semester = models.PositiveSmallIntegerField()
    minor = models.ForeignKey(Minor, null=True, blank=True, on_delete=models.PROTECT, related_name="+")
    minor_registered = models.BooleanField(default=False)
    # B.E. code picked on the profile when the ID is M.Sc.-only (dual degrees are allotted after the 1st year, so
    # the ID may not show it yet); turns programme into the dual one. Blank when the ID already has it.
    second_degree_code = models.CharField(max_length=2, blank=True, default="")
    interests = models.JSONField(default=list, blank=True)
    # "X or Y" chart slots: {"ECON F211": "MGTS F211"} = took MGTS F211 in the slot the chart lists as ECON F211
    alternative_choices = models.JSONField(default=dict, blank=True)
    # optional extras for ranking
    cgpa = models.DecimalField(max_digits=4, decimal_places=2, null=True, blank=True)
    goal = models.CharField(max_length=30, choices=GOALS, blank=True, default="")
    strengths = models.TextField(blank=True, default="")
    weaknesses = models.TextField(blank=True, default="")
    sop_plan = models.BooleanField(null=True, blank=True)
    # recommender defaults; a chat message can override them ("8 AM is fine")
    default_avoid_8am = models.BooleanField(default=False)
    default_avoid_day = models.CharField(max_length=2, choices=[(code, DAY_NAMES[code]) for code in DAY_CODES],
                                         null=True, blank=True, default=None)
    avoid_eval_styles = models.JSONField(default=list, blank=True, validators=[validate_eval_styles])

    def __str__(self) -> str:
        return f"{self.user} ({self.programme})"

    def save(self, *args, **kwargs) -> None:
        validate_eval_styles(self.avoid_eval_styles)  # why: JSONField validators only run in full_clean, not save
        super().save(*args, **kwargs)


class StudentCourse(models.Model):
    STATUS = [("completed", "completed"), ("current", "current")]
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name="courses")
    course = models.ForeignKey(Course, on_delete=models.PROTECT, related_name="+")
    status = models.CharField(max_length=10, choices=STATUS)
    grade = models.CharField(max_length=3, choices=GRADES, null=True, blank=True)
    comfort = models.CharField(max_length=1, choices=COMFORT, blank=True, default="")  # electives only
    # CDC / GIR / DEL / HUEL / OPEL / AUDIT for this student's programme, set when the row is saved
    # (recommender/categories.py). Stored so "done courses by category" is one query; recomputed on every save.
    category = models.CharField(max_length=10, blank=True, default="")
    semester_taken = models.CharField(max_length=30, blank=True, default="")
    counted_as = models.CharField(max_length=20, blank=True, default="")  # optional category override
    source = models.CharField(max_length=10, choices=[("pattern", "pattern"), ("user", "user")], default="user")
    pattern_slot = models.ForeignKey(PatternSlot, null=True, blank=True, on_delete=models.PROTECT, related_name="+")
    needs_review = models.BooleanField(default=False)  # pre-filled but uncertain (e.g. first-year variants)

    class Meta:
        unique_together = ("student", "course")


class SemesterPlan(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name="plans")
    semester_tag = models.CharField(max_length=30)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("student", "semester_tag")


class PlanItem(models.Model):
    ADDED_BY = [("user", "user"), ("agent", "agent")]
    plan = models.ForeignKey(SemesterPlan, on_delete=models.CASCADE, related_name="items")
    course = models.ForeignKey(Course, on_delete=models.PROTECT, related_name="+")
    lecture_section = models.ForeignKey(Section, null=True, blank=True, on_delete=models.PROTECT, related_name="+")
    tutorial_section = models.ForeignKey(Section, null=True, blank=True, on_delete=models.PROTECT, related_name="+")
    practical_section = models.ForeignKey(Section, null=True, blank=True, on_delete=models.PROTECT, related_name="+")
    added_by = models.CharField(max_length=5, choices=ADDED_BY, default="user")


class TimetablePreference(models.Model):
    student = models.OneToOneField(Student, on_delete=models.CASCADE, related_name="timetable_preference")
    no_8am_days = models.JSONField(default=list, blank=True)  # ["M", "W"]
    free_day = models.CharField(max_length=2, blank=True, default="")
    no_saturday = models.BooleanField(default=False)
    avoid_gaps = models.BooleanField(default=False)
    compact = models.BooleanField(default=False)
    priority_order = models.JSONField(default=list, blank=True)


class ChatState(models.Model):
    student = models.OneToOneField(Student, on_delete=models.CASCADE, related_name="chat_state")
    requests = models.JSONField(default=list, blank=True)  # [{"category": "DEL", "want": ["AI"], "dont_want": ["midsem"]}]
    messages = models.JSONField(default=list, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
