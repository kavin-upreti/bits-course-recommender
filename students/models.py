"""Per-student data (ideation Part 5, 'Student data'). Not from PDFs; never touched by `manage.py ingest`.

why PROTECT on catalog FKs: ingest clears and reloads the catalog; PROTECT makes it refuse instead of silently
deleting students' courses and plans (see the ingest command's --wipe-students).
"""
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models

from catalog.models import Course, Minor, PatternSlot, Programme
from recommender.config import DAY_CODES, DAY_NAMES

MAX_PICKED_COURSES = 5  # "did well in" / "struggled with": a few telling courses, not a grade for every course
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
    strengths = models.TextField(blank=True, default="")
    # completed course codes the student picked; courses close to them in content rank higher / lower
    did_well = models.JSONField(default=list, blank=True)
    struggled = models.JSONField(default=list, blank=True)
    grade_oriented = models.BooleanField(default=False)  # the "did well" boost counts double
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
