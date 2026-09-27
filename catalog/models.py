"""Course, programme and rule data (ideation Part 5). Filled only by `manage.py ingest`.

`sources` = {field_name: {doc, section, page, quote}}; `needs_verification` + `note` flag anything uncertain.
"""
from django.db import models


class Flagged(models.Model):
    """Common provenance fields for every record read from a PDF or curated by hand."""

    sources = models.JSONField(default=dict, blank=True)
    needs_verification = models.BooleanField(default=False)
    note = models.TextField(blank=True, default="")

    class Meta:
        abstract = True


# ---------------------------------------------------------------- course data

class Course(Flagged):
    """One course code. From the timetable (offered this semester) and/or the Bulletin (all courses)."""

    code = models.CharField(max_length=20, unique=True)  # "CS F211"
    title = models.CharField(max_length=200, blank=True, default="")
    department = models.CharField(max_length=10)  # code prefix: "CS"
    L = models.PositiveSmallIntegerField(null=True, blank=True)
    P = models.PositiveSmallIntegerField(null=True, blank=True)
    T = models.PositiveSmallIntegerField(null=True, blank=True)
    S = models.PositiveSmallIntegerField(null=True, blank=True)
    units = models.PositiveSmallIntegerField(null=True, blank=True)
    com_code = models.PositiveIntegerField(null=True, blank=True)
    only_2026_batch = models.BooleanField(default=False)  # com_code >= 5000
    description = models.TextField(blank=True, default="")
    prerequisites = models.JSONField(null=True, blank=True)  # AND of OR-groups: [["CE F231", "ME F212"], ["MATH F211"]]
    is_project_course = models.BooleanField(default=False)  # number matches the Bulletin's XXX F266/F366/... patterns
    # ponytail: "G" level letter = higher degree (BITS numbering); not stated as a rule in our PDFs (ideation Part 10)
    is_higher_degree = models.BooleanField(default=False)
    llm_tags = models.JSONField(null=True, blank=True)  # filled later (ideation 4.7)

    def __str__(self) -> str:
        return f"{self.code} {self.title}"


class CourseEquivalent(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="equivalents")
    equivalent_code = models.CharField(max_length=20)
    # timetable / bulletin: listed in the source documents; content: found by recommender/equivalents.py
    source = models.CharField(max_length=10, default="timetable")

    class Meta:
        unique_together = ("course", "equivalent_code")


class Offering(Flagged):
    """A course in one semester's timetable. A code can appear more than once (different com codes)."""

    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="offerings")
    semester_tag = models.CharField(max_length=30)  # "2026-27 Sem 1"
    campus = models.CharField(max_length=30)
    com_code = models.PositiveIntegerField(null=True, blank=True)
    midsem = models.CharField(max_length=30, blank=True, default="")
    midsem_date = models.CharField(max_length=10, blank=True, default="")
    midsem_session = models.CharField(max_length=10, blank=True, default="")
    compre = models.CharField(max_length=30, blank=True, default="")
    compre_date = models.CharField(max_length=10, blank=True, default="")
    compre_session = models.CharField(max_length=10, blank=True, default="")
    instructor_in_charge = models.CharField(max_length=200, blank=True, default="")


class Section(models.Model):
    TYPES = [("lecture", "lecture"), ("tutorial", "tutorial"), ("practical", "practical")]
    offering = models.ForeignKey(Offering, on_delete=models.CASCADE, related_name="sections")
    section_id = models.CharField(max_length=10)  # "L1"
    type = models.CharField(max_length=10, choices=TYPES)
    instructors = models.JSONField(default=list)
    room = models.CharField(max_length=50, blank=True, default="")
    timings = models.JSONField(default=dict)  # {"M": [1, 2]}
    cancelled = models.BooleanField(default=False)


class Handout(Flagged):
    """One handout PDF (a course can have several). Exam dates / textbooks / consultation hours are not stored."""

    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="handouts")
    file = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, default="")  # the handout's course description paragraph
    objectives = models.JSONField(default=list)  # ["To understand the Laplace transform ...", ...]
    learning_outcomes = models.JSONField(default=list)  # what the student will be able to do, one item each
    topics = models.JSONField(default=list)  # flat, for search
    topic_groups = models.JSONField(default=list)  # [{"module": str | None, "topics": [str]}], for display
    evaluation = models.JSONField(default=list)
    has_midsem = models.BooleanField(null=True)
    open_book_percent = models.FloatField(null=True, blank=True)
    has_project = models.BooleanField(null=True)
    project_percent = models.FloatField(null=True, blank=True)
    has_quiz = models.BooleanField(null=True)
    compre_percent = models.FloatField(null=True, blank=True)
    attendance_text = models.TextField(blank=True, default="")
    attendance_required = models.BooleanField(null=True)
    attendance_percent = models.PositiveSmallIntegerField(null=True, blank=True)
    attendance_follows_default = models.BooleanField(default=False)
    makeup_text = models.TextField(blank=True, default="")
    makeup_allowed = models.BooleanField(null=True)
    makeup_per_component = models.JSONField(default=dict)
    makeup_follows_default = models.BooleanField(default=False)
    embedding = models.JSONField(null=True, blank=True)  # filled later for interest matching


class CoursePiece(models.Model):
    """One short piece of text about a course (title, a description sentence, a topic, an outcome) and its embedding.
    Built by `manage.py build_embeddings` (recommender/pieces.py); a course is matched by its best pieces."""

    KINDS = [(kind, kind) for kind in ("title", "description", "topic", "outcome", "bulletin_description")]
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="pieces")  # FK = indexed
    kind = models.CharField(max_length=25, choices=KINDS)
    text = models.TextField()
    source = models.CharField(max_length=100)  # handout file name, "bulletin", or "timetable" for the title
    embedding = models.BinaryField()  # float32 bytes, L2-normalised
    model_name = models.CharField(max_length=100)


# ---------------------------------------------------------------- programme data

class Programme(Flagged):
    TYPES = [("single", "single"), ("dual", "dual"), ("dual_template", "dual template")]
    name = models.CharField(max_length=200, unique=True)
    degree = models.CharField(max_length=30, blank=True, default="")
    type = models.CharField(max_length=15, choices=TYPES)
    discipline_code = models.CharField(max_length=10, blank=True, default="")  # "CS"; blank for dual (see components)
    first_component = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    second_component = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    edition = models.CharField(max_length=20, blank=True, default="")
    batch_range = models.CharField(max_length=50, null=True, blank=True)  # not stated in the Bulletin
    core_units = models.PositiveSmallIntegerField(null=True, blank=True)
    core_courses = models.PositiveSmallIntegerField(null=True, blank=True)
    del_units = models.PositiveSmallIntegerField(null=True, blank=True)
    del_courses = models.PositiveSmallIntegerField(null=True, blank=True)
    summer = models.JSONField(null=True, blank=True)  # PS-I
    final_year_options = models.JSONField(default=list, blank=True)  # PS-II / thesis

    def __str__(self) -> str:
        return self.name


class PatternSlot(models.Model):
    """One line of a programme's semester-wise chart."""

    programme = models.ForeignKey(Programme, on_delete=models.CASCADE, related_name="slots")
    year = models.PositiveSmallIntegerField()
    semester = models.PositiveSmallIntegerField()
    slot_type = models.CharField(max_length=10)  # named / elective
    course = models.ForeignKey(Course, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    alternatives = models.JSONField(default=list)
    elective_category = models.CharField(max_length=30, blank=True, default="")
    units_text = models.CharField(max_length=20, blank=True, default="")
    semester_unit_total = models.CharField(max_length=20, blank=True, default="")


class ProgrammeCourse(models.Model):
    """Which category a course is for a programme (CDC / DEL / project-DEL)."""

    programme = models.ForeignKey(Programme, on_delete=models.CASCADE, related_name="courses")
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="programme_links")
    category = models.CharField(max_length=15)  # CDC / DEL / project-DEL
    component = models.PositiveSmallIntegerField(default=1)  # dual degrees: 1st or 2nd degree's list
    track_or_pool = models.CharField(max_length=100, blank=True, default="")
    alternative_group = models.JSONField(default=list)  # codes that can replace this one
    inferred = models.BooleanField(default=False)  # e.g. BIOT -> BIO by title (code_mappings)
    sources = models.JSONField(default=dict, blank=True)


class GirCourse(models.Model):
    """A named General Institutional Requirement course (category_requirements.json); classifies foundation courses."""

    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="gir")
    heading = models.CharField(max_length=100)  # Science Foundation / Technical Arts / ...
    alternative_group = models.JSONField(default=list)
    sources = models.JSONField(default=dict, blank=True)


class HuelPoolCourse(models.Model):
    course = models.OneToOneField(Course, on_delete=models.CASCADE, related_name="huel")


class AuditCourse(models.Model):
    code = models.CharField(max_length=20, unique=True)
    title = models.CharField(max_length=200, blank=True, default="")


class Minor(Flagged):
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True, default="")
    min_courses = models.PositiveSmallIntegerField(null=True, blank=True)
    min_units = models.PositiveSmallIntegerField(null=True, blank=True)
    exclusion_text = models.TextField(blank=True, default="")

    def __str__(self) -> str:
        return self.name


class MinorCourse(models.Model):
    minor = models.ForeignKey(Minor, on_delete=models.CASCADE, related_name="courses")
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="+")
    role = models.CharField(max_length=10)  # core / elective


# ---------------------------------------------------------------- rules (hand-curated)

class CategoryRequirement(Flagged):
    category = models.CharField(max_length=100)
    code = models.CharField(max_length=20, blank=True, default="")  # "HUEL"
    group = models.CharField(max_length=100, blank=True, default="")
    min_units = models.PositiveSmallIntegerField(null=True, blank=True)
    max_units = models.PositiveSmallIntegerField(null=True, blank=True)
    min_courses = models.PositiveSmallIntegerField(null=True, blank=True)
    max_courses = models.PositiveSmallIntegerField(null=True, blank=True)
    named_courses = models.JSONField(default=list)


class Rule(Flagged):
    rule_id = models.CharField(max_length=100, unique=True)
    group = models.CharField(max_length=30)  # regulations / dual_degree / minor / project / huel / timetable
    description = models.TextField()
    values = models.JSONField(default=dict)
    quote = models.TextField(blank=True, default="")
    source_doc = models.CharField(max_length=100, blank=True, default="")
    applies_to_batches = models.CharField(max_length=100, null=True, blank=True)


class KnownGap(Flagged):
    gap_id = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    affected = models.JSONField(default=dict)
    user_message = models.TextField(null=True, blank=True)


class CodeMapping(Flagged):
    """Hand-inferred code mapping, e.g. the Bulletin's BIOT F211 -> the timetable's BIO F211 (same title)."""

    from_code = models.CharField(max_length=20, unique=True)
    to_course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name="mapped_from")
    inferred = models.BooleanField(default=True)
    reason = models.TextField(blank=True, default="")
