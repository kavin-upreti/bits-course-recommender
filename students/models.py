"""Per-student data (ideation Part 5, 'Student data'). Not from PDFs; never touched by `manage.py ingest`.

why PROTECT on catalog FKs: ingest clears and reloads the catalog; PROTECT makes it refuse instead of silently
deleting students' courses and plans (see the ingest command's --wipe-students).
"""
from django.conf import settings
from django.db import models

from catalog.models import Course, Minor, Programme, Section


class Student(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="student")
    campus = models.CharField(max_length=30, default="Pilani")
    admission_year = models.PositiveSmallIntegerField()
    programme = models.ForeignKey(Programme, on_delete=models.PROTECT, related_name="+")
    second_programme = models.ForeignKey(Programme, null=True, blank=True, on_delete=models.PROTECT, related_name="+")
    current_year = models.PositiveSmallIntegerField()
    current_semester = models.PositiveSmallIntegerField()
    minor = models.ForeignKey(Minor, null=True, blank=True, on_delete=models.PROTECT, related_name="+")
    minor_registered = models.BooleanField(default=False)
    interests = models.JSONField(default=list, blank=True)
    # optional extras for ranking
    cgpa_by_semester = models.JSONField(default=dict, blank=True)
    goal = models.CharField(max_length=30, blank=True, default="")  # job / research / not sure
    strengths = models.TextField(blank=True, default="")
    weaknesses = models.TextField(blank=True, default="")
    sop_plan = models.BooleanField(null=True, blank=True)

    def __str__(self) -> str:
        return f"{self.user} ({self.programme})"


class StudentCourse(models.Model):
    STATUS = [("completed", "completed"), ("current", "current")]
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name="courses")
    course = models.ForeignKey(Course, on_delete=models.PROTECT, related_name="+")
    status = models.CharField(max_length=10, choices=STATUS)
    grade = models.CharField(max_length=3, null=True, blank=True)
    semester_taken = models.CharField(max_length=30, blank=True, default="")
    counted_as = models.CharField(max_length=20, blank=True, default="")  # optional category override

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
