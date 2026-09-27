"""check_plan (todo.md 7.6, tests 1-14)."""
from unittest.mock import patch

from recommender import config
from recommender.context import build_context
from recommender.tools import check_plan

from .fixtures import RecommenderTestCase, base_rules, make_course, make_programme, make_student, offer, take


class CheckPlanTests(RecommenderTestCase):
    def setUp(self) -> None:
        super().setUp()
        base_rules()
        self.student = make_student(make_programme())
        self.current = self.course("CC F101", [("lecture", "L1", {"M": [2], "W": [2]}), ("practical", "P1", {"Th": [7, 8]})],
                                   midsem=("10/10", "FN1"), compre=("10/12", "FN"))
        take(self.student, self.current, "current")

    def course(self, code: str, sections: list[tuple], units: int = 3, midsem=("", ""), compre=("", "")):
        course = make_course(code, units=units)
        offer(course, sections, midsem, compre)
        return course

    def plan(self, codes: list[str], **kwargs) -> dict:
        return check_plan(build_context(self.student), codes, **kwargs)

    def test_01_two_courses_fit(self):
        self.course("AA F201", [("lecture", "L1", {"T": [3]}), ("tutorial", "T1", {"F": [9]})], midsem=("11/10", "FN1"))
        self.course("BB F202", [("lecture", "L1", {"T": [4]})], midsem=("12/10", "FN1"))
        result = self.plan(["aa f201", "BB F202"])
        self.assertEqual(result, {"ok": True, "sections": {"AA F201": {"lecture": "L1", "tutorial": "T1"},
                                                           "BB F202": {"lecture": "L1"}},
                                  "includes_current_courses": ["CC F101"], "total_units": 9,
                                  "notes": ["AA F201's compre slot isn't in the timetable, so its compre clashes weren't checked",
                                            "BB F202's compre slot isn't in the timetable, so its compre clashes weren't checked"]})

    def test_one_higher_degree_course_per_semester(self):
        from catalog.models import Rule
        Rule.objects.create(rule_id="reg_higher_degree_course", group="regulations", description="", values={"max_per_semester": 1})
        for code, day in (("AA G511", "T"), ("BB G512", "F")):
            course = self.course(code, [("lecture", "L1", {day: [3]})])
            course.is_higher_degree = True
            course.save()
        self.assertTrue(self.plan(["AA G511"])["ok"])
        result = self.plan(["AA G511", "BB G512"])
        self.assertEqual(result["problem"], "Too many higher degree courses: 2 higher degree courses (AA G511, BB G512), at most 1 per semester.")

    def test_02_alternative_section_picked(self):
        self.course("AA F201", [("lecture", "L1", {"M": [2]}), ("lecture", "L2", {"T": [3]})])
        self.assertEqual(self.plan(["AA F201"])["sections"], {"AA F201": {"lecture": "L2"}})

    def test_03_and_10_lab_second_period_clash_with_current_course(self):
        self.course("AA F201", [("lecture", "L1", {"Th": [8]})])
        result = self.plan(["AA F201"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["conflicts"], [{"a": "AA F201", "b": "CC F101", "type": "class", "detail": "L1 and P1 on Th, period 8"}])
        self.assertEqual(result["problem"], "AA F201 clashes with CC F101 in every section combination (e.g. L1 and P1 on Th, period 8).")
        self.assertEqual(result["includes_current_courses"], ["CC F101"])

    def test_04_lunch_rule(self):
        self.course("AA F201", [("lecture", "L1", {"M": [4, 5]})])
        self.course("BB F202", [("lecture", "L1", {"M": [6]})])
        result = self.plan(["AA F201", "BB F202"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["conflicts"][0], {"a": "AA F201", "b": "BB F202", "type": "lunch",
                                                  "detail": "together they fill periods 4, 5 and 6 on M"})

    def test_05_exam_clashes(self):
        self.course("AA F201", [("lecture", "L1", {"T": [3]})], midsem=("10/10", "FN1"))
        result = self.plan(["AA F201"])
        self.assertEqual(result["conflicts"][0], {"a": "CC F101", "b": "AA F201", "type": "midsem", "detail": "both on 10/10, FN1"})
        self.assertEqual(result["problem"], "CC F101 and AA F201 have their midsem at the same time (both on 10/10, FN1).")
        self.course("BB F202", [("lecture", "L1", {"T": [4]})], midsem=("11/10", "FN1"), compre=("10/12", "FN"))
        self.assertEqual([c["type"] for c in self.plan(["BB F202"])["conflicts"]], ["compre"])

    def test_06_units(self):
        self.course("AA F201", [("lecture", "L1", {"T": [3]})], units=23)
        result = self.plan(["AA F201"])
        self.assertEqual(result["conflicts"], [{"a": "all courses", "b": "units", "type": "units", "detail": "total 26 units, maximum is 25"}])
        new = make_student(make_programme("B.E. New", "NN"), "2026A7PS0001P", 2026)
        result = check_plan(build_context(new), ["AA F201"])
        self.assertTrue(result["ok"])
        self.assertIn("maximum units for your batch isn't in the data, so it wasn't checked", result["notes"])

    def test_07_cancelled_never_chosen(self):
        self.course("AA F201", [("lecture", "L1", {"T": [3]}, True), ("lecture", "L2", {"T": [4]})])
        self.assertEqual(self.plan(["AA F201"])["sections"]["AA F201"], {"lecture": "L2"})

    def test_08_avoid_8am(self):
        self.course("AA F201", [("lecture", "L1", {"T": [1]}), ("lecture", "L2", {"T": [3]})])
        self.assertEqual(self.plan(["AA F201"], avoid_8am=True)["sections"]["AA F201"], {"lecture": "L2"})
        self.course("BB F202", [("lecture", "L1", {"F": [1], "S": [1]})])
        result = self.plan(["BB F202"], avoid_8am=True)
        self.assertTrue(result["ok"])
        self.assertIn("no section combination avoids 8 AM for BB F202; chose L1 (8 AM on F S)", result["notes"])

    def test_09_avoid_day(self):
        self.course("AA F201", [("lecture", "L1", {"F": [3]}), ("lecture", "L2", {"T": [3]})])
        self.assertEqual(self.plan(["AA F201"], avoid_day="F")["sections"]["AA F201"], {"lecture": "L2"})
        self.course("BB F202", [("lecture", "L1", {"F": [4]})])
        result = self.plan(["BB F202"], avoid_day="F")
        self.assertIn("no section combination keeps Friday free for BB F202; chose L1 (class on Friday)", result["notes"])
        self.student.default_avoid_day = "T"  # profile default used when the message doesn't say
        self.student.save()
        self.assertEqual(self.plan(["AA F201"])["sections"]["AA F201"], {"lecture": "L1"})

    def test_11_fits_in_pairs_but_not_together(self):
        for code in ("AA F201", "BB F202", "DD F203"):
            self.course(code, [("lecture", "L1", {"T": [2]}), ("lecture", "L2", {"T": [3]})])
        result = self.plan(["AA F201", "BB F202", "DD F203"])
        self.assertEqual(result, {"ok": False, "problem": "Each pair of courses fits, but not all of them together.",
                                  "conflicts": [], "includes_current_courses": ["CC F101"]})

    def test_12_unknown_not_offered_and_current(self):
        self.assertEqual(self.plan(["NOPE F999"]), {"ok": False, "problem": "No course with code NOPE F999"})
        make_course("AA F201")
        self.assertEqual(self.plan(["AA F201"]), {"ok": False, "problem": "AA F201 isn't offered this semester"})
        result = self.plan(["CC F101"])
        self.assertEqual((result["ok"], result["sections"]), (True, {}))
        self.assertEqual(self.plan([])["problem"], "Give between 1 and 10 course codes.")

    def test_current_course_not_in_timetable(self):
        take(self.student, make_course("EE F301"), "current")
        self.course("AA F201", [("lecture", "L1", {"T": [3]})])
        self.assertIn("EE F301 (current) isn't in this semester's timetable, so it wasn't checked", self.plan(["AA F201"])["notes"])

    def test_untimed_section_note(self):
        self.course("AA F201", [("lecture", "L1", {})])
        self.assertIn("class times for AA F201 L1 aren't listed, so clashes with it couldn't be checked", self.plan(["AA F201"])["notes"])

    def test_13_node_limit(self):
        self.course("AA F201", [("lecture", "L1", {"T": [1]}), ("lecture", "L2", {"T": [3]})])
        self.course("BB F202", [("lecture", "L1", {"T": [3]})])
        self.course("DD F203", [("lecture", "L1", {"S": [2]}), ("lecture", "L2", {"S": [3]})])
        with patch.object(config, "PLAN_SEARCH_NODE_LIMIT", 1):
            unknown = self.plan(["AA F201", "BB F202", "DD F203"])
            self.assertEqual(unknown["problem"], "too many combinations to check; try fewer courses")
            self.assertTrue(unknown["limit_hit"])  # callers treat it as unknown, not as a clash
        with patch.object(config, "PLAN_SEARCH_NODE_LIMIT", 6):  # finds one 8 AM plan, then hits the limit
            result = self.plan(["AA F201", "BB F202", "DD F203"], avoid_8am=True)
        self.assertTrue(result["ok"])
        self.assertIn("search limit reached; this is a valid timetable but maybe not the best one", result["notes"])

    def test_14_deterministic(self):
        for code, tutorial in (("AA F201", 8), ("BB F202", 9)):
            self.course(code, [("lecture", "L1", {"T": [2]}), ("lecture", "L2", {"T": [3]}), ("tutorial", "T1", {"F": [tutorial]})])
        first = self.plan(["BB F202", "AA F201"], avoid_8am=True)
        self.assertEqual(self.plan(["BB F202", "AA F201"], avoid_8am=True), first)
        self.assertEqual(list(first["sections"]), ["BB F202", "AA F201"])
