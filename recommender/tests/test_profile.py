"""Profile boost (eligible.apply_profile): interests, strengths, did well / struggled with. Fake embedder + reranker."""
from unittest.mock import patch

from recommender import config
from recommender.context import build_context
from recommender.eligible import profile_phrases, split_branches

from .fixtures import make_course, take
from .test_eligible import Catalog

# XX F412 Deep Learning sits next to the "neural networks" texts; the other DELs point elsewhere
VECTORS = {"Natural Language Processing": [1, 0, 0], "Deep Learning": [0, 1, 0], "Compiler Construction": [0, 0, 1],
           "Neural Networks": [0.1, 1, 0], "neural networks": [0.1, 1, 0]}


class ProfileBoostTests(Catalog):
    def setUp(self) -> None:
        super().setUp()
        self.done = make_course("XX F111", "Neural Networks")
        take(self.student, self.done)
        self.embed(VECTORS)

    def update(self, **fields) -> None:
        for name, value in fields.items():
            setattr(self.student, name, value)
        self.student.save()

    def test_did_well_and_struggled_reorder_without_a_topic(self):
        self.assertEqual(self.codes(self.result(category="DEL")), ["XX F411", "XX F412", "XX F413"])  # by code
        self.update(did_well=["XX F111"])
        result = self.result(category="DEL")
        self.assertEqual(self.codes(result)[0], "XX F412")
        self.assertIn("close to XX F111 Neural Networks, which you did well in", result["courses"][0]["score"]["why"])
        self.update(did_well=[], struggled=["XX F111"])
        result = self.result(category="DEL")
        self.assertEqual(self.codes(result)[-1], "XX F412")
        self.assertIn("which you struggled with", result["courses"][-1]["score"]["why"])

    def test_grade_oriented_doubles_the_did_well_part(self):
        self.update(did_well=["XX F111"])
        normal = self.result(category="DEL")["courses"][0]["score"]["final"]
        self.update(grade_oriented=True)
        self.assertAlmostEqual(self.result(category="DEL")["courses"][0]["score"]["final"], 2 * normal, delta=0.011)

    def test_strengths_text_boosts_close_courses(self):
        self.update(strengths="neural networks")
        result = self.result(category="DEL")
        self.assertEqual(self.codes(result)[0], "XX F412")
        self.assertIn("matches your strength 'neural networks'", result["courses"][0]["score"]["why"])

    def test_strengths_are_split_into_phrases(self):
        self.update(interests=["robotics"], strengths="programming and algorithms, statistics; design & drawing")
        phrases = [phrase for phrase, _ in profile_phrases(build_context(self.student), "about")]
        self.assertEqual(phrases, ["robotics", "programming", "algorithms", "statistics", "design", "drawing"])
        # interests that are already the query aren't counted again
        self.assertEqual(profile_phrases(build_context(self.student), "profile_interests")[0][0], "programming")

    def test_a_course_not_done_is_ignored(self):
        self.update(did_well=["XX F413"])  # offered, not completed: a stale pick doesn't count
        self.assertEqual(self.codes(self.result(category="DEL")), ["XX F411", "XX F412", "XX F413"])

    def test_with_a_topic_the_boost_never_adds_a_course(self):
        self.reranker.scores.update({"Natural Language Processing": 0.9, "Deep Learning": 0.1, "Compiler Construction": 0.1})
        self.update(did_well=["XX F111"], grade_oriented=True, strengths="neural networks")
        self.assertEqual(self.codes(self.result(category="DEL", about="language")), ["XX F411"])

    def test_personal_is_in_the_score_only_when_it_counts(self):
        self.assertNotIn("personal", self.result(category="DEL")["courses"][0]["score"])
        self.update(did_well=["XX F111"])
        score = self.result(category="DEL")["courses"][0]["score"]
        self.assertGreater(score["personal"], 0)
        self.assertAlmostEqual(score["final"], score["personal"] + score["penalty"], delta=0.011)

    def test_sop_plan_reminder(self):
        self.assertFalse(any("SOP" in warning for warning in self.result(category="DEL")["warnings"]))
        self.update(sop_plan=True)
        self.assertTrue(any("keep a slot free" in warning for warning in self.result(category="DEL")["warnings"]))


class BranchTests(Catalog):
    def test_split_branches(self):
        self.assertEqual(split_branches(["maths"]), (["MATH"], []))
        self.assertEqual(split_branches(["Maths courses", "probability"]), (["MATH"], ["probability"]))
        self.assertEqual(split_branches(["electrical engineering", "finance"]), (["EEE", "FIN", "ECON"], []))
        self.assertEqual(split_branches(["financial markets", "maths for machine learning"]),
                         ([], ["financial markets", "maths for machine learning"]))
        self.assertEqual(split_branches(None), ([], []))

    def test_branch_limits_the_search(self):
        with patch.dict(config.BRANCH_ALIASES, {"humanities": ["HSS"], "soil": ["ZZ"]}):
            result = self.result(about=["humanities courses"])
            self.assertEqual(sorted(self.codes(result)), ["HSS F201", "HSS F202"])
            self.assertEqual(result["settings_used"]["branches"], ["HSS"])
            self.assertEqual(result["settings_used"]["ranked_by"], "handout_completeness")  # no topic left, no interests
            result = self.result(category="DEL", about=["soil"])
            self.assertEqual(result["courses"], [])
            self.assertIn("No DELs from ZZ are offered this semester.", result["warnings"])


class ProjectCourseTests(Catalog):
    def test_projects_rank_after_lecture_courses(self):
        project = self.courses["XX F411"]
        project.is_project_course = True
        project.save()
        self.assertEqual(self.codes(self.result(category="DEL")), ["XX F412", "XX F413", "XX F411"])


class BranchStrengthTests(Catalog):
    def test_a_branch_word_strength_boosts_that_department(self):
        with patch.dict(config.BRANCH_ALIASES, {"humanities": ["HSS"]}):
            self.student.strengths = "humanities"
            self.student.save()
            result = self.result()
        top = result["courses"][0]
        self.assertTrue(top["code"].startswith("HSS "))
        self.assertIn("matches your strength 'humanities'", top["score"]["why"])
