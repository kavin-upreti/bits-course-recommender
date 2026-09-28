"""handout_facts: every rule, on unsaved Handout objects (no database needed)."""
from django.test import SimpleTestCase

from catalog.models import Handout
from recommender.handout_facts import compute_course_facts, makeup_text, quiz_component_count


def component(kind: str, weight: float = 10, name: str = "", nature: str | None = None) -> dict:
    return {"name": name or kind.title(), "kind": kind, "weightage_percent": weight, "nature": nature}


def handout(file: str = "1_XX_F101.pdf", evaluation: list | None = None, **fields) -> Handout:
    return Handout(file=file, evaluation=evaluation or [], **fields)


FULL = [component("midsem", 30, nature="CB"), component("compre", 40, nature="OB"),
        component("quiz", 10, "Quizzes 1-3"), component("project", 20)]


class QuizCountTests(SimpleTestCase):
    def test_names(self):
        cases = {
            "Quiz": 1, "Quiz 2": 1, "Quiz/Test": 1, "Quiz 1-3": 3, "Quiz I–III": 3, "Quizzes (3)": 3, "3 Quizzes": 3,
            "Quiz (One)": 1, "Quizzes 1 & 2": 2, "Quizzes 1, 2 and 3": 3, "Announced Quizzes (Best 5 out of 6)": 6,
            "Lecture Quiz (3 out of 4)": 4, "3 Class tests (Best 2)": 3, "Quiz (4 quizzes during lecture hours)": 4,
            "Quizzes": None, "Class Quizzes": None, "Quiz(zes) / Assignment(s)": None, "Class Tests": None,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(quiz_component_count(name), expected)


class CourseFactsTests(SimpleTestCase):
    def test_no_handout(self):
        facts = compute_course_facts("XX F101", [])
        self.assertFalse(facts.evaluation_parsed)
        self.assertEqual(facts.notes, {"all": "no handout"})
        self.assertIsNone(facts.has_midsem)
        self.assertEqual(facts.completeness, 0)
        self.assertEqual(facts.note_for("has_midsem"), "no handout")
        self.assertEqual(facts.unknown_text("has_midsem"), "couldn't verify (no handout)")

    def test_single_handout(self):
        facts = compute_course_facts("XX F101", [handout(evaluation=FULL, attendance_required=True, makeup_allowed=True,
                                                         makeup_per_component={"midsem": True, "compre": True, "quiz": False})])
        self.assertTrue(facts.evaluation_parsed)
        self.assertEqual((facts.has_midsem, facts.quiz_count, facts.compre_percent, facts.project_percent, facts.open_book),
                         (True, 3, 40, 20, True))
        self.assertEqual((facts.attendance_required, facts.makeup_allowed, facts.makeup_lenient), (True, True, True))
        self.assertEqual(facts.sources, ["1_XX_F101.pdf"])
        self.assertEqual(facts.completeness, 8)

    def test_parsed_without_components_is_false_or_zero(self):
        facts = compute_course_facts("XX F101", [handout(evaluation=[component("compre", 100, nature="CB")])])
        self.assertEqual((facts.has_midsem, facts.quiz_count, facts.project_percent, facts.open_book), (False, 0, 0, False))

    def test_not_parsed(self):
        facts = compute_course_facts("XX F101", [handout(attendance_required=None, attendance_follows_default=True)])
        self.assertFalse(facts.evaluation_parsed)
        for name in ("has_midsem", "quiz_count", "compre_percent", "project_percent", "open_book"):
            self.assertIsNone(getattr(facts, name))
        self.assertEqual(facts.note_for("has_midsem"), "handout doesn't list its evaluation")
        self.assertEqual(facts.note_for("attendance_required"), "follows institute rules")
        self.assertEqual(facts.unknown_text("makeup_allowed"), "couldn't verify")

    def test_open_book_from_percent(self):
        facts = compute_course_facts("XX F101", [handout(evaluation=[component("compre", 100)], open_book_percent=20)])
        self.assertTrue(facts.open_book)

    def test_plural_quiz_without_number(self):
        facts = compute_course_facts("XX F101", [handout(evaluation=[component("quiz", 20, "Quizzes"), component("quiz", 5, "Quiz 1")])])
        self.assertIsNone(facts.quiz_count)
        self.assertEqual(facts.notes["quiz_count"], "quiz count not stated")

    def test_makeup_lenient(self):
        def lenient(allowed, per_component):
            return compute_course_facts("XX F101", [handout(makeup_allowed=allowed, makeup_per_component=per_component)]).makeup_lenient
        self.assertIs(lenient(False, {}), False)
        self.assertIs(lenient(True, {"midsem": True, "compre": False}), False)
        self.assertIs(lenient(None, {"midsem": False}), False)
        self.assertIs(lenient(True, {"quiz": False}), True)  # quizzes don't matter
        self.assertIsNone(lenient(None, {"midsem": True}))

    def test_multiple_agreeing(self):
        first = handout("1_XX_F101.pdf", FULL, attendance_required=True)
        second = handout("2_XX_F101.pdf", FULL, attendance_required=None)
        facts = compute_course_facts("XX F101", [second, first])
        self.assertEqual((facts.quiz_count, facts.attendance_required), (3, True))
        self.assertEqual(facts.sources, ["1_XX_F101.pdf", "2_XX_F101.pdf"])
        self.assertNotIn("attendance_required", facts.notes)

    def test_multiple_differing(self):
        first = handout("1_XX_F101.pdf", FULL)
        second = handout("2_XX_F101.pdf", [component("compre", 60), component("quiz", 40, "Quiz")])
        facts = compute_course_facts("XX F101", [first, second])
        self.assertIsNone(facts.has_midsem)
        self.assertIsNone(facts.quiz_count)
        self.assertEqual(facts.notes["has_midsem"], "handouts differ")
        self.assertEqual(facts.note_for("quiz_count"), "handouts differ")
        self.assertEqual(facts.project_percent, None)

    def test_parsed_and_unparsed_handouts(self):
        facts = compute_course_facts("XX F101", [handout("1_XX_F101.pdf", FULL), handout("2_XX_F101.pdf", [])])
        self.assertTrue(facts.evaluation_parsed)
        self.assertEqual(facts.compre_percent, 40)


class MakeupFromTextTests(SimpleTestCase):
    """Makeup text the extractor left unclassified (allowed None) is read at fact time."""

    def facts(self, text: str):
        return compute_course_facts("X", [handout(makeup_text=text)])

    def test_conditional_grant_means_makeups_exist(self):
        facts = self.facts("To be granted only in case of serious illness or emergency. Plagiarism Policy: reported.")
        self.assertIs(facts.makeup_allowed, True)
        self.assertEqual(makeup_text(facts), "only on conditions: “To be granted only in case of serious illness or emergency.”")

    def test_institute_rules_whole_course_refusal_and_one_component(self):
        self.assertEqual(makeup_text(self.facts("As per AUGSD guidelines")), "follows institute rules")
        self.assertIs(self.facts("No makeup for this course. Social Conduct").makeup_allowed, False)
        # a refusal for one component says nothing about the course as a whole
        self.assertIsNone(self.facts("Make-up will not be given for the tutorial tests.").makeup_allowed)

    def test_extractor_decision_wins(self):
        facts = compute_course_facts("X", [handout(makeup_text="Only in genuine cases", makeup_allowed=False)])
        self.assertIs(facts.makeup_allowed, False)
