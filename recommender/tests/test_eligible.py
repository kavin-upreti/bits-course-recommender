"""get_remaining_requirements, get_course_details (todo 5.2 / 5.3) and get_eligible_courses (todo 6.8, tests 1-20)."""
import math
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.db import transaction

from catalog.models import HuelPoolCourse, Programme
from recommender import config
from recommender.context import build_context
from recommender.eligible import run
from recommender.handout_facts import clear_facts_cache
from recommender.tools import get_course_details, get_eligible_courses, get_remaining_requirements

from .fixtures import (
    RecommenderTestCase, add_handout, base_rules, equivalent, link, make_course, make_programme, make_student,
    minor_with, offer, take,
)

EIGHT_AM = {"M": [1], "W": [1]}
LATE = {"T": [3], "Th": [3]}


def with_sim(similarity: float) -> list[float]:
    """A 2-d unit vector with the given dot product against the query [1, 0]."""
    return [similarity, math.sqrt(1 - similarity ** 2)]


class Catalog(RecommenderTestCase):
    """B.E. Test (XX): DELs XX F411-F413, HUELs HSS F201-F202, OPELs YY F301-F302, CDC XX F211 (current)."""

    def setUp(self) -> None:
        super().setUp()
        base_rules()
        self.programme = make_programme()
        self.courses = {}
        for code, title, category in [
            ("XX F411", "Natural Language Processing", "DEL"), ("XX F412", "Deep Learning", "DEL"),
            ("XX F413", "Compiler Construction", "DEL"), ("HSS F201", "Introductory Psychology", "HUEL"),
            ("HSS F202", "Film Studies", "HUEL"), ("YY F301", "Financial Markets", "OPEL"),
            ("YY F302", "Soil Science", "OPEL"), ("XX F211", "Data Structures", "CDC"),
        ]:
            course = self.courses[code] = make_course(code, title)
            # the current CDC meets on its own hour, so every elective fits with it
            offer(course, [("lecture", "L1", {"M": [9]} if code == "XX F211" else LATE)])
            if category in ("DEL", "CDC"):
                link(self.programme, course, category)
            if category == "HUEL":
                HuelPoolCourse.objects.create(course=course)
        self.student = make_student(self.programme)
        take(self.student, self.courses["XX F211"], "current")
        call_command("build_embeddings", stdout=StringIO())  # every course has at least its title piece

    def result(self, **kwargs) -> dict:
        return get_eligible_courses(build_context(self.student), **kwargs)

    def codes(self, result: dict, key: str = "courses") -> list[str]:
        return [course["code"] for course in result[key]]

    def embed(self, vectors: dict[str, list[float]]) -> None:
        """Give texts fixed vectors, then (re)build the pieces with them."""
        self.embedder.set(vectors)
        call_command("build_embeddings", stdout=StringIO())

    def dual_student(self):
        """M.Sc.-style dual of B.E. Test + B.E. Other (ZZ), whose DEL ZZ F411 has no handout."""
        second = make_programme("B.E. Other", "ZZ")
        dual = Programme.objects.create(name="Dual", type="dual", first_component=self.programme, second_component=second)
        other_del = make_course("ZZ F411", "Robotics")
        offer(other_del)
        link(second, other_del)
        link(dual, other_del, component=2)
        for code in ("XX F411", "XX F412", "XX F413"):
            link(dual, self.courses[code])
        return make_student(dual, "2025B5A4PS0001P", 2025)

    def fill(self, category: str, count: int, status: str = "completed") -> None:
        """Mark `count` new courses of a category as taken."""
        for number in range(count):
            course = make_course(f"{'ZZ' if category == 'OPEL' else 'XX' if category == 'DEL' else 'HSS'} F{700 + number}{category[0]}")
            if category == "DEL":
                link(self.programme, course)
            if category == "HUEL":
                HuelPoolCourse.objects.create(course=course)
            take(self.student, course, status)


class RequirementsAndDetailsTests(Catalog):
    def test_remaining_single(self):
        take(self.student, self.courses["HSS F201"])
        self.assertEqual(get_remaining_requirements(build_context(self.student)), {
            "HUEL": {"remaining_courses": 0, "remaining_units": 0, "complete": True},
            "DEL": {"remaining_courses": 2, "remaining_units": 6, "complete": False},
            "OPEL": {"remaining_courses": 5, "remaining_units": 15, "complete": False},
        })

    def test_remaining_dual_adds_dels(self):
        dual = self.dual_student()
        result = get_remaining_requirements(build_context(dual))
        self.assertEqual(result["DEL"], {"remaining_courses": 4, "remaining_units": 12, "complete": False})
        self.assertEqual(result["note"], "dual degree: DEL counts are for both degrees combined")

    def test_remaining_known_gap(self):
        new = make_student(self.programme, "2026A7PS0001P", 2026)
        self.assertEqual(get_remaining_requirements(build_context(new)),
                         {"error": "Your batch follows the new curriculum, so requirements can't be calculated."})

    def test_details(self):
        course = self.courses["XX F411"]
        course.description = "Bulletin text. Second sentence. Third sentence."
        course.prerequisites = [["XX F211"], ["MATH F111", "MATH F112"]]
        course.save()
        add_handout(course, file="a.pdf", description="Handout first. Handout second. Handout third.",
                    attendance_required=True, makeup_allowed=True, makeup_per_component={"midsem": True, "compre": True})
        add_handout(course, file="b.pdf", evaluation=[{"name": "Compre", "kind": "compre", "weightage_percent": 100, "nature": "CB"}])
        details = get_course_details(build_context(self.student), "xx  f411")
        self.assertEqual(details, {
            "code": "XX F411", "title": "Natural Language Processing", "category": "DEL", "units": 3,
            "prerequisites": ["XX F211", "MATH F111 or MATH F112"], "prerequisites_met": False,
            "offered_this_semester": True,
            "facts": {"midsem": "couldn't verify (handouts differ)", "quizzes": "couldn't verify (handouts differ)",
                      "compre_percent": "couldn't verify (handouts differ)", "project_percent": 0, "open_book": "no",
                      "attendance": "required", "makeup": "allowed for midsem and compre"},
            "summary": "Handout first. Handout second.",
        })

    def test_details_no_handout_dual_and_2026(self):
        details = get_course_details(build_context(self.dual_student()), "ZZ F411")
        self.assertEqual((details["category"], details["summary"]), ("DEL", ""))
        self.assertEqual(details["facts"]["midsem"], "couldn't verify (no handout)")
        new = make_student(self.programme, "2026A7PS0001P", 2026)
        self.assertEqual(get_course_details(build_context(new), "YY F301")["category"], "OPEL")
        self.assertEqual(get_course_details(build_context(new), "NOPE F999"), {"error": "No course with code NOPE F999"})


class StageATests(Catalog):
    def test_01_category_filter_and_dual(self):
        self.assertEqual(self.codes(self.result(category="DEL")), ["XX F411", "XX F412", "XX F413"])
        self.assertEqual(self.codes(self.result(category="HUEL")), ["HSS F201", "HSS F202"])
        dual = self.dual_student()
        result = get_eligible_courses(build_context(dual), category="DEL")
        self.assertIn("ZZ F411", self.codes(result))  # the second degree's DEL
        self.assertIn("ZZ F411", self.codes(self.result(category="OPEL")))  # single degree: just an OPEL

    def test_02_done_equivalent_and_current_removed(self):
        take(self.student, self.courses["XX F411"])
        old = make_course("XX F499", "Deep Learning (old code)")
        equivalent(self.courses["XX F412"], "XX F499")
        take(self.student, old)
        take(self.student, self.courses["XX F413"], "current")
        result, pipeline = run(build_context(self.student), "DEL", None, None, None, None, None)
        self.assertEqual(result["courses"], [])
        self.assertEqual(sorted(pipeline.removed["done"]), ["XX F411", "XX F412", "XX F413"])
        self.assertIn("No DELs you haven't already taken are offered this semester.", result["warnings"])

    def test_03_prerequisite_via_current_course(self):
        compiler = self.courses["XX F413"]
        compiler.prerequisites = [["XX F211"]]
        compiler.save()
        result = self.result(category="DEL")
        self.assertNotIn("XX F413", self.codes(result))
        self.assertIn("XX F413 needs XX F211, which you're taking this semester, so you'll be eligible next semester",
                      result["warnings"])

    def test_04_unparseable_prerequisite_kept(self):
        compiler = self.courses["XX F413"]
        compiler.needs_verification, compiler.note = True, "prerequisite text has no course codes"
        compiler.save()
        entry = next(course for course in self.result(category="DEL")["courses"] if course["code"] == "XX F413")
        self.assertEqual(entry["note"], "prerequisites couldn't be verified")

    def test_05_newest_batch_only_course(self):
        new_course = make_course("XX F414", "New Curriculum Course", only_2026_batch=True)
        offer(new_course)
        link(self.programme, new_course)
        self.assertNotIn("XX F414", self.codes(self.result(category="DEL")))
        new = make_student(self.programme, "2026A7PS0001P", 2026)
        self.assertIn("XX F414", self.codes(get_eligible_courses(build_context(new), category="DEL")))

    def test_06_exclude(self):
        equivalent(self.courses["XX F412"], "XX F499")
        result = self.result(category="DEL", exclude=["xx f411", "XX F499", "NOPE F999"])
        self.assertEqual(self.codes(result), ["XX F413"])
        self.assertEqual(result["warnings"], ["Unknown course in exclude: NOPE F999"])

    def test_higher_degree_note_and_minor(self):
        higher = make_course("XX G511", "Advanced Topics", is_higher_degree=True)
        offer(higher)
        link(self.programme, higher)
        self.student.minor = minor_with("Test minor", [], [self.courses["XX F412"]])
        self.student.save()
        entries = {course["code"]: course for course in self.result(category="DEL")["courses"]}
        self.assertEqual(entries["XX G511"]["note"], "higher-degree course: at most 1 per semester, needs a minimum CGPA set by the AGC (the number isn't published)")
        self.assertEqual(entries["XX F412"]["minor"], "elective")
        self.assertNotIn("minor", entries["XX F411"])


class FilterTests(Catalog):
    CASES = {
        # filter: (value, pass handout, fail handout, unknown handout, unknown note)
        "no_midsem": (True, {"evaluation": [{"name": "Compre", "kind": "compre", "weightage_percent": 100, "nature": "CB"}]},
                      {}, {"evaluation": []}, "handout doesn't list its evaluation"),
        "no_attendance": (True, {"attendance_required": False}, {"attendance_required": True},
                          {"attendance_required": None, "attendance_follows_default": True}, "follows institute rules"),
        "lenient_makeup": (True, {"makeup_allowed": True, "makeup_per_component": {"midsem": True, "compre": True}},
                           {"makeup_allowed": False}, {"makeup_allowed": None}, "not stated in the handout"),
        "project_based": (True, {"evaluation": [{"name": "Project", "kind": "project", "weightage_percent": 20, "nature": None}]},
                          {}, None, "no handout"),
        "open_book": (True, {"evaluation": [{"name": "Compre", "kind": "compre", "weightage_percent": 100, "nature": "OB"}]},
                      {}, {"evaluation": []}, "handout doesn't list its evaluation"),
        "max_quizzes": (2, {}, {"evaluation": [{"name": "Quizzes 1-3", "kind": "quiz", "weightage_percent": 30, "nature": "CB"}]},
                        {"evaluation": [{"name": "Quizzes", "kind": "quiz", "weightage_percent": 30, "nature": "CB"}]},
                        "quiz count not stated"),
        "max_compre_percent": (40, {}, {"evaluation": [{"name": "Compre", "kind": "compre", "weightage_percent": 50, "nature": "CB"}]},
                               None, "no handout"),
        "min_project_percent": (20, {"evaluation": [{"name": "Project", "kind": "project", "weightage_percent": 25, "nature": None}]},
                                {"evaluation": [{"name": "Project", "kind": "project", "weightage_percent": 10, "nature": None}]},
                                None, "no handout"),
    }

    def test_07_each_filter_pass_fail_unknown(self):
        for number, (key, case) in enumerate(self.CASES.items()):
            with self.subTest(filter=key), transaction.atomic():
                self.check_filter(number, key, *case)
                transaction.set_rollback(True)  # why: each filter sees only its own three courses

    def check_filter(self, number: int, key: str, value, passing: dict, failing: dict, unknown: dict | None, note: str) -> None:
        codes = [f"QQ F{number}0{outcome}" for outcome in (1, 2, 3)]
        for code, fields in zip(codes, (passing, failing, unknown)):
            course = make_course(code, f"Filter course {code}")
            offer(course)
            link(self.programme, course)
            if fields is not None:
                add_handout(course, **fields)
        result = self.result(category="DEL", filters={key: value})
        self.assertIn(codes[0], self.codes(result))
        self.assertNotIn(codes[1], self.codes(result) + self.codes(result, "couldnt_verify"))
        unverified = {course["code"]: course for course in result["couldnt_verify"]}
        self.assertEqual(unverified[codes[2]]["unverified"], [key])
        self.assertIn(note, unverified[codes[2]]["note"])
        passed = next(course for course in result["courses"] if course["code"] == codes[0])
        self.assertEqual(passed["filters_passed"], [key])

    def test_false_boolean_filter_is_not_applied(self):
        self.assertEqual(self.codes(self.result(category="DEL", filters={"no_midsem": False})), ["XX F411", "XX F412", "XX F413"])

    def test_08_couldnt_verify_is_capped_counted_and_ranked(self):
        similarities = [0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8]
        vectors = {"neural networks": [1, 0], "Natural Language Processing": with_sim(-0.5), "Deep Learning": with_sim(-0.5),
                   "Compiler Construction": with_sim(-0.5)}
        for number, similarity in enumerate(similarities):
            course = make_course(f"QQ F9{number}0", f"Unknown Course {number}")
            offer(course)
            link(self.programme, course)
            vectors[f"Unknown Course {number}"] = with_sim(similarity)
        self.embed(vectors)
        result = self.result(category="DEL", about="neural networks", filters={"no_midsem": True})
        # 7 + the 3 fixture DELs are unverified; the best MAX_UNVERIFIED are returned, the rest counted
        self.assertEqual([course["code"] for course in result["couldnt_verify"]],
                         ["QQ F910", "QQ F960", "QQ F940", "QQ F920", "QQ F930"])
        self.assertEqual(result["couldnt_verify_more"], 2)  # the fixture DELs (relevance 0.25) fell below the cutoff


class RankingTests(Catalog):
    def test_09_about_then_interests_then_completeness(self):
        self.embed({"compilers": [1, 0], "Compiler Construction": [1, 0], "Deep Learning": [0, 1],
                    "neural networks": [0, 1]})
        result = self.result(category="DEL", about="compilers")
        self.assertEqual((result["settings_used"]["ranked_by"], self.codes(result)[0]), ("about", "XX F413"))
        self.student.interests = ["neural networks"]
        self.student.save()
        result = self.result(category="DEL")
        self.assertEqual((result["settings_used"]["ranked_by"], self.codes(result)[0]), ("profile_interests", "XX F412"))
        self.assertTrue(result["courses"][0]["score"]["why"].startswith("related to your profile interests"))
        self.student.interests = []
        self.student.save()
        add_handout(self.courses["XX F413"], attendance_required=True, makeup_allowed=True)
        clear_facts_cache()  # handouts only change on ingest, which clears it
        result = self.result(category="DEL")
        self.assertEqual(result["settings_used"]["ranked_by"], "handout_completeness")
        self.assertEqual(self.codes(result), ["XX F413", "XX F411", "XX F412"])
        self.assertIsNone(result["courses"][0]["score"]["relevance"])

    def test_11_dislike_penalties_and_why(self):
        add_handout(self.courses["XX F411"], evaluation=[
            {"name": "Quizzes 1-4", "kind": "quiz", "weightage_percent": 20, "nature": "CB"},
            {"name": "Compre", "kind": "compre", "weightage_percent": 50, "nature": "CB"},
        ], attendance_required=True, makeup_allowed=False)
        self.student.avoid_eval_styles = ["many_quizzes", "closed_book", "strict_attendance", "heavy_compre", "no_makeup"]
        self.student.save()
        self.embed({"nlp": [1, 0], "Natural Language Processing": [1, 0], "Deep Learning": [0.1, 1]})
        entries = {c["code"]: c for c in self.result(category="DEL", about="nlp")["courses"]}
        score = entries["XX F411"]["score"]
        self.assertEqual((score["relevance"], score["penalty"], score["final"]), (1.0, -0.25, 0.75))
        self.assertEqual(score["why"], "matches title: Natural Language Processing; 4 quizzes (you'd rather avoid many quizzes); "
                         "no open-book exams (you'd rather avoid closed-book exams); attendance required (you'd rather avoid "
                         "strict attendance); compre is 50% (you'd rather avoid a heavy compre); no makeups (you'd rather "
                         "avoid courses without makeups)")
        self.assertEqual(entries["XX F412"]["score"]["penalty"], 0.0)  # no handout: unknown facts never penalise
        self.assertIn("no handout, matched on the Bulletin description only", entries["XX F412"]["score"]["why"])

    def test_12_tie_break_category_then_code(self):
        self.embed({"anything": [1, 0], "Natural Language Processing": [1, 0], "Deep Learning": [1, 0],
                    "Introductory Psychology": [1, 0], "Financial Markets": [1, 0], "Film Studies": [0, 1],
                    "Soil Science": [0, 1], "Compiler Construction": [0, 1]})
        result = self.result(about="anything")
        self.assertEqual(self.codes(result)[:4], ["XX F411", "XX F412", "YY F301", "HSS F201"])


class TimetablePreferenceTests(Catalog):
    def build(self, sections_by_code: dict[str, list[tuple]]) -> None:
        """Replace sections of the fixture DELs, and add filler DELs ranked above them."""
        for code, sections in sections_by_code.items():
            self.courses[code].offerings.all().delete()
            offer(self.courses[code], sections)

    def test_13_avoid_8am(self):
        self.build({
            "XX F411": [("lecture", "L1", EIGHT_AM), ("lecture", "L2", LATE)],               # one lecture avoids 8 AM: kept
            "XX F412": [("lecture", "L1", EIGHT_AM), ("tutorial", "T1", {"F": [1]}), ("tutorial", "T2", LATE)],  # every lecture at 8
            "XX F413": [("lecture", "L1", LATE), ("practical", "P1", {"F": [1, 2]})],         # only practical at 8
        })
        result = self.result(category="DEL", avoid_8am=True)
        self.assertEqual(self.codes(result), ["XX F411"])
        self.assertEqual({(entry["code"], entry["reason"]) for entry in result["excluded"]},
                         {("XX F412", "every lecture section has an 8 AM class"),
                          ("XX F413", "every practical section has an 8 AM class")})
        self.assertEqual(set(result["excluded"][0]), {"code", "title", "category", "reason"})

    def test_13_excluded_only_if_in_top_five(self):
        vectors = {"topic": [1, 0], "Compiler Construction": with_sim(-0.9), "Natural Language Processing": with_sim(0.5),
                   "Deep Learning": with_sim(0.5)}
        for number in range(6):
            course = make_course(f"XX F5{number}0", f"Filler {number}")
            offer(course, [("lecture", "L1", LATE)])
            link(self.programme, course)
            vectors[f"Filler {number}"] = with_sim(0.9 - number / 100)
        self.build({"XX F413": [("lecture", "L1", EIGHT_AM)]})
        self.embed(vectors)
        result = self.result(category="DEL", about="topic", avoid_8am=True)
        self.assertNotIn("XX F413", self.codes(result))
        self.assertEqual(result["excluded"], [])  # rank 9 before stage D

    def test_14_avoid_day(self):
        self.build({
            "XX F411": [("lecture", "L1", {"F": [3]}), ("lecture", "L2", LATE)],
            "XX F412": [("lecture", "L1", LATE), ("tutorial", "T1", {"F": [5]})],
            "XX F413": [("lecture", "L1", {"M": [1], "F": [2]})],
        })
        result = self.result(category="DEL", avoid_day="F", avoid_8am=True)
        self.assertEqual(self.codes(result), ["XX F411"])
        reasons = {entry["code"]: entry["reason"] for entry in result["excluded"]}
        self.assertEqual(reasons, {"XX F412": "every tutorial section is on Friday",
                                   "XX F413": "every lecture section has an 8 AM class or is on Friday"})

    def test_untimed_sections_are_ok_with_note(self):
        self.build({"XX F411": [("lecture", "L1", {})]})
        entry = next(c for c in self.result(category="DEL", avoid_8am=True)["courses"] if c["code"] == "XX F411")
        self.assertEqual(entry["note"], "some class times aren't listed in the timetable")

    def test_15_profile_defaults_and_sources(self):
        self.build({"XX F412": [("lecture", "L1", EIGHT_AM)]})
        self.student.default_avoid_8am, self.student.default_avoid_day = True, "S"
        self.student.save()
        result = self.result(category="DEL")
        self.assertNotIn("XX F412", self.codes(result))
        self.assertEqual(result["settings_used"], {"avoid_8am": True, "avoid_8am_source": "profile", "avoid_day": "S",
                                                   "avoid_day_source": "profile", "ranked_by": "handout_completeness"})
        result = self.result(category="DEL", avoid_8am=False)
        self.assertIn("XX F412", self.codes(result))
        self.assertEqual((result["settings_used"]["avoid_8am"], result["settings_used"]["avoid_8am_source"]), (False, "message"))


class RequirementStateTests(Catalog):
    def test_16_no_category_searches_incomplete_ones(self):
        take(self.student, self.courses["HSS F201"])
        result = self.result()
        self.assertEqual(result["searched"], ["DEL", "OPEL"])
        # HUEL is complete, so a HUEL only shows up as an OPEL
        entry = next(course for course in result["courses"] if course["code"] == "HSS F202")
        self.assertEqual((entry["category"], entry["counts_as"]), ("HUEL", "OPEL"))
        self.assertEqual(entry["note"], "counts as an OPEL, since your HUEL requirement is complete")

    def test_16_all_complete(self):
        self.fill("HUEL", 1)
        self.fill("DEL", 2)
        self.fill("OPEL", 5)
        result = self.result()
        self.assertEqual((result["searched"], result["courses"]), ([], []))
        self.assertEqual(result["warnings"], ["Your HUEL, DEL and OPEL requirements are all complete. Any further elective "
                                              "would be an extra elective (the regulations allow at most 4 extra)."])

    def test_17_opel_includes_dels_once_del_is_complete(self):
        self.fill("DEL", 2)
        result = self.result(category="OPEL")
        entry = next(course for course in result["courses"] if course["code"] == "XX F411")
        self.assertEqual((entry["category"], entry["counts_as"], entry["note"]),
                         ("DEL", "OPEL", "counts as an OPEL, since your DEL requirement is complete"))
        self.assertNotIn("HSS F201", self.codes(result))  # HUEL isn't complete

    def test_18_explicit_complete_category(self):
        self.fill("DEL", 2)
        result = self.result(category="DEL")
        self.assertEqual(self.codes(result), ["XX F411", "XX F412", "XX F413"])
        self.assertEqual(result["warnings"], ["Your DEL requirement is already complete (6 of 6 units). "
                                              "Extra DELs will count as OPELs instead."])
        self.fill("OPEL", 5)
        self.assertEqual(self.result(category="DEL")["warnings"], [
            "Your DEL requirement is already complete (6 of 6 units). Extra DELs will be extra electives (at most 4 extra allowed)."])

    def test_19_known_gap(self):
        new = make_student(self.programme, "2026A7PS0001P", 2026)
        result = get_eligible_courses(build_context(new))
        self.assertEqual(result["searched"], ["HUEL", "DEL", "OPEL"])
        self.assertEqual(result["warnings"], ["Your batch follows the new curriculum, so requirements can't be calculated."])

    def test_20_shape_cap_and_determinism(self):
        for number in range(8):
            course = make_course(f"XX F6{number}0", f"Extra {number}")
            offer(course)
            link(self.programme, course)
        self.embed({})  # pieces for the new courses
        patch.object(config, "RELEVANCE_CUTOFF", 0.0).start()  # hash-vector scores: keep everything for this test
        self.addCleanup(patch.stopall)
        first = self.result(category="DEL", about="anything at all")
        self.assertEqual(list(first), ["searched", "settings_used", "courses", "couldnt_verify", "excluded", "warnings"])
        self.assertEqual(len(first["courses"]), config.MAX_RESULTS)
        allowed = {"code", "title", "category", "counts_as", "minor", "note", "filters_passed", "sections", "score"}
        for entry in first["courses"]:
            self.assertLessEqual(set(entry), allowed)
            self.assertEqual(set(entry["score"]), {"embedding", "relevance", "penalty", "final", "why", "topic"})
        self.assertEqual(self.result(category="DEL", about="anything at all"), first)
        self.assertEqual(len(self.result(category="DEL", about="anything at all", count=7)["courses"]), 7)  # "suggest 7"


class TopicQualityTests(Catalog):
    """Weak-match warning, better matches that aren't offered, and topic-filtered "next semester" notes."""

    def test_better_match_not_offered(self):
        unoffered = make_course("XX F499", "System Security")  # a DEL with no offering this semester
        link(self.programme, unoffered)
        self.embed({"security": [1, 0], "System Security": [1, 0], "Natural Language Processing": with_sim(0.3),
                    "Deep Learning": with_sim(0.2), "Compiler Construction": with_sim(0.1)})
        result = self.result(category="DEL", about="security")
        self.assertEqual(result["better_matches_not_offered"], [{"code": "XX F499", "title": "System Security", "category": "DEL"}])

    def test_good_match_has_no_warning_or_extras(self):
        self.embed({"nlp": [1, 0], "Natural Language Processing": [1, 0]})
        result = self.result(category="DEL", about="nlp")
        self.assertNotIn("better_matches_not_offered", result)
        self.assertFalse(any("strongly matches" in warning for warning in result["warnings"]))

    def test_next_semester_notes_only_for_relevant_courses(self):
        for code in ("XX F412", "XX F413"):
            self.courses[code].prerequisites = [["XX F211"]]  # XX F211 is current
            self.courses[code].save()
        self.embed({"neural networks": [1, 0], "Deep Learning": [1, 0], "Compiler Construction": [-1, 0]})
        warnings = self.result(category="DEL", about="neural networks")["warnings"]
        self.assertIn("XX F412 needs XX F211, which you're taking this semester, so you'll be eligible next semester", warnings)
        self.assertFalse(any(warning.startswith("XX F413") for warning in warnings))  # unrelated to the topic


class FitTests(Catalog):
    def test_courses_that_cannot_fit_current_ones_are_excluded(self):
        self.courses["XX F412"].offerings.all().delete()
        offer(self.courses["XX F412"], [("lecture", "L1", {"M": [9]})])  # the current XX F211's only hour
        result = self.result(category="DEL")
        self.assertEqual(self.codes(result), ["XX F411", "XX F413"])
        self.assertEqual(result["courses"][0]["sections"], {"lecture": "L1"})
        self.assertEqual(result["excluded"], [{"code": "XX F412", "title": "Deep Learning", "category": "DEL",
                                               "reason": "doesn't fit with your current courses: XX F412 clashes with XX F211 "
                                                         "in every section combination (e.g. both L1 on M, period 9)."}])


    def test_8am_free_sections_must_fit_with_current_courses(self):
        self.courses["XX F411"].offerings.all().delete()
        # L2 avoids 8 AM but meets at the current XX F211's hour; only the 8 AM L1 fits
        offer(self.courses["XX F411"], [("lecture", "L1", {"T": [1]}), ("lecture", "L2", {"M": [9]})])
        result = self.result(category="DEL", avoid_8am=True)
        self.assertNotIn("XX F411", self.codes(result))
        self.assertIn({"code": "XX F411", "title": "Natural Language Processing", "category": "DEL",
                       "reason": "every way it fits with your current courses has an 8 AM class"}, result["excluded"])
        self.assertIn("XX F411", self.codes(self.result(category="DEL", avoid_8am=False)))
