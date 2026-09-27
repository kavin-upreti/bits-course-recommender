"""A small handmade catalog for recommender tests: fast, and independent of the real data.

Programme "B.E. Test" (discipline XX) needs 1 HUEL (3 units), 2 DELs (6 units) and 5 OPELs (15 units): the OPEL
count comes from recommender.requirements.opel_requirement with 16 core courses / 54 core units.
"""
from django.contrib.auth.models import User
from django.test import TestCase

from catalog.models import (
    CategoryRequirement, Course, CourseEquivalent, Handout, HuelPoolCourse, KnownGap, Minor, MinorCourse, Offering,
    Programme, ProgrammeCourse, Rule, Section,
)
from unittest.mock import patch

from recommender import config, embeddings, handout_facts, piece_index
from recommender.testing import FakeEmbedder, FakeReranker
from students.models import Student, StudentCourse

SEMESTER = "2026-27 Sem 1"
EVALUATION = [
    {"name": "Midsem", "kind": "midsem", "weightage_percent": 30, "nature": "CB"},
    {"name": "Compre", "kind": "compre", "weightage_percent": 40, "nature": "CB"},
    {"name": "Quiz", "kind": "quiz", "weightage_percent": 30, "nature": "CB"},
]


def make_course(code: str, title: str = "", units: int | None = 3, **fields) -> Course:
    return Course.objects.create(code=code, title=title or f"Course {code}", department=code.split()[0], units=units, **fields)


def offer(course: Course, sections: list[tuple] | None = None, midsem: tuple[str, str] = ("", ""),
          compre: tuple[str, str] = ("", "")) -> Offering:
    """sections: (type, id, timings) or (type, id, timings, cancelled). Default: one lecture, M W F period 2."""
    offering = Offering.objects.create(course=course, semester_tag=SEMESTER, campus="Pilani",
                                       midsem_date=midsem[0], midsem_session=midsem[1],
                                       compre_date=compre[0], compre_session=compre[1])
    for section in sections if sections is not None else [("lecture", "L1", {"M": [2], "W": [2], "F": [2]})]:
        kind, section_id, timings, *cancelled = section
        Section.objects.create(offering=offering, type=kind, section_id=section_id, timings=timings,
                               cancelled=bool(cancelled and cancelled[0]))
    return offering


def add_handout(course: Course, evaluation: list | None = None, file: str = "", **fields) -> Handout:
    return Handout.objects.create(course=course, file=file or f"{course.code.replace(' ', '_')}.pdf",
                                  evaluation=EVALUATION if evaluation is None else evaluation, **fields)


def link(programme: Programme, course: Course, category: str = "DEL", component: int = 1) -> None:
    ProgrammeCourse.objects.create(programme=programme, course=course, category=category, component=component)


def make_programme(name: str = "B.E. Test", discipline: str = "XX") -> Programme:
    return Programme.objects.create(name=name, type="single", discipline_code=discipline, core_courses=16, core_units=54,
                                    del_courses=2, del_units=6)


def base_rules() -> None:
    """Requirements, regulations and the 2026 known gap, shaped like the real ingested rows."""
    CategoryRequirement.objects.create(category="Humanities Electives", code="HUEL", min_courses=1, min_units=3)
    CategoryRequirement.objects.create(category="Open Electives", code="OPEL", min_courses=5, min_units=15)
    Rule.objects.create(rule_id="reg_max_units", group="regulations", description="", values={"first_degree_max_units": 25})
    Rule.objects.create(rule_id="reg_max_units_2026", group="regulations", description="", values={"first_degree_max_units": None})
    Rule.objects.create(rule_id="reg_extra_electives", group="regulations", description="", values={"max_extra_electives": 4})
    Rule.objects.create(rule_id="minor_units", group="minor", description="",
                        values={"total_min_courses": 5, "electives_min_courses": 2})
    KnownGap.objects.create(gap_id="gap_new", description="", affected={"batches": ["2026 onwards"]},
                            user_message="Your batch follows the new curriculum, so requirements can't be calculated.")


def make_student(programme: Programme, username: str = "2024A7PS0001P", admission_year: int = 2024, **fields) -> Student:
    user = User.objects.create_user(username=username, password="x", first_name="Asha Testname")
    return Student.objects.create(user=user, programme=programme, admission_year=admission_year, current_year=3,
                                  current_semester=1, **fields)


def take(student: Student, course: Course, status: str = "completed") -> None:
    StudentCourse.objects.create(student=student, course=course, status=status, source="user")


def equivalent(course: Course, code: str) -> None:
    CourseEquivalent.objects.create(course=course, equivalent_code=code)


def minor_with(name: str, core: list[Course], electives: list[Course]) -> Minor:
    minor = Minor.objects.create(name=name, min_courses=5)
    for role, courses in (("core", core), ("elective", electives)):
        for course in courses:
            MinorCourse.objects.create(minor=minor, course=course, role=role)
    return minor


class RecommenderTestCase(TestCase):
    """Fresh caches per test (handout facts, piece index) and a fake embedder the test can give vectors to."""

    def setUp(self) -> None:
        # why pinned: the tests check behaviour; the tuned values (EDA) may change without breaking them
        # NEIGHBOUR_ANCHOR_FLOOR above 1: neighbours off (the fake reranker scores any pair ~0.5); NeighbourTests turn them on
        for name, value in (("RELEVANCE_CUTOFF", 0.5), ("NEIGHBOUR_ANCHOR_FLOOR", 1.1)):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        handout_facts.clear_facts_cache()
        piece_index.invalidate_piece_index()
        self.embedder = FakeEmbedder()
        embeddings.set_embedder(self.embedder)
        self.reranker = FakeReranker()
        embeddings.set_reranker(self.reranker)
        self.addCleanup(embeddings.set_embedder, FakeEmbedder())
        self.addCleanup(embeddings.set_reranker, FakeReranker())
        self.addCleanup(handout_facts.clear_facts_cache)
        self.addCleanup(piece_index.invalidate_piece_index)
