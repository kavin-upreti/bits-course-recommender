"""Retrieve-then-rerank ranking with a relevance cutoff (todo.md 6.4, tests 1-9). Fake embedder + fake reranker."""
import json
from unittest.mock import patch

import numpy as np

from django.test import SimpleTestCase

from recommender import config
from recommender.agent import run_agent
from recommender.embeddings import SentenceTransformerEmbedder
from recommender.handout_facts import clear_facts_cache
from recommender.llm import LLMResponse, ToolCall
from recommender.ranking import Scored, rerank
from recommender.testing import FakeLLM

from .fixtures import add_handout, equivalent, link, make_course, offer
from .test_eligible import Catalog

TITLES = ["Natural Language Processing", "Deep Learning", "Compiler Construction"]


class RerankTests(Catalog):
    def scores(self, **by_text: float) -> None:
        """Reranker scores per piece text; every DEL title not given scores 0.1."""
        self.reranker.scores.update({title: 0.1 for title in TITLES})
        self.reranker.scores.update(by_text)

    def test_01_one_passing_mention_doesnt_carry_a_course(self):
        strong, mild = self.courses["XX F411"], self.courses["XX F412"]
        strong.description = "Network security and cryptography."
        strong.save()
        mild.description = "Some systems. Some networks. Some protocols. Some servers. Some clients."
        mild.save()
        self.embed({})
        self.scores(**{"Network security and cryptography.": 0.95, "Natural Language Processing": 0.1, "Deep Learning": 0.6,
                       **{f"Some {word}.": 0.6 for word in ("systems", "networks", "protocols", "servers", "clients")}})
        result = self.result(category="DEL", about="cybersecurity")
        # relevance = mean of the best 3 pieces: steady content (0.6) beats one strong line next to a weak title
        self.assertEqual([course["code"] for course in result["courses"]], ["XX F412", "XX F411"])
        self.assertEqual(result["courses"][0]["score"]["relevance"], 0.6)
        self.assertAlmostEqual(result["courses"][1]["score"]["relevance"], (0.95 + 0.1) / 2, delta=0.006)
        self.assertIn("matches bulletin description: Network security and cryptography.", result["courses"][1]["score"]["why"])

    def test_02_cutoff_no_padding(self):
        self.scores(**{"Natural Language Processing": 0.9})
        result = self.result(category="DEL", about="language")
        self.assertEqual([course["code"] for course in result["courses"]], ["XX F411"])  # fewer than 5: no padding
        self.assertFalse(any("directly" in warning for warning in result["warnings"]))

        self.scores(**{"Natural Language Processing": 0.45, "Deep Learning": 0.4})  # Compiler Construction 0.1
        result = self.result(category="DEL", about="marine biology")
        self.assertEqual(result["courses"], [])
        self.assertIn("Nothing this semester matches 'marine biology' directly.", result["warnings"])

    def test_03_no_cutoff_for_interests_or_completeness(self):
        self.scores(**{"Deep Learning": 0.6})  # one related course: ranked by the interests, the rest still shown
        self.student.interests = ["gardening"]
        self.student.save()
        result = self.result(category="DEL")
        self.assertEqual((result["settings_used"]["ranked_by"], len(result["courses"])), ("profile_interests", 3))
        self.student.interests = []
        self.student.save()
        calls = len(self.reranker.calls)
        result = self.result(category="DEL")
        self.assertEqual((result["settings_used"]["ranked_by"], len(result["courses"])), ("handout_completeness", 3))
        self.assertEqual(len(self.reranker.calls), calls)  # no reranking without a query

    def test_unrelated_interests_fall_back_to_handout_order(self):
        self.scores()  # every DEL title 0.1: nothing relates to the interests
        self.student.interests = ["machine learning"]
        self.student.save()
        result = self.result(category="DEL")
        self.assertEqual(result["settings_used"]["ranked_by"], "handout_completeness")
        self.assertEqual(len(result["courses"]), 3)
        self.assertIsNone(result["courses"][0]["score"]["relevance"])

    def test_a_course_named_after_the_topic_beats_one_that_mentions_it(self):
        teaches, uses = self.courses["XX F411"], self.courses["XX F412"]
        teaches.description = "N-gram models. Part of speech tagging."
        teaches.save()
        uses.description = "Applications in natural language processing. Text models for NLP."
        uses.save()
        self.embed({})
        self.scores(**{"Natural Language Processing": 0.9, "N-gram models.": 0.0, "Part of speech tagging.": 0.0,
                       "Applications in natural language processing.": 0.95, "Text models for NLP.": 0.9})
        courses = self.result(category="DEL", about="nlp")["courses"]
        # the old blend gave the named course 0.5 * 0.9 + 0.5 * 0.3 = 0.6 (cut off) and the user 0.8
        self.assertEqual([course["code"] for course in courses], ["XX F411", "XX F412"])
        self.assertEqual(courses[0]["score"]["relevance"], 0.9)
        self.assertIn("matches title: Natural Language Processing", courses[0]["score"]["why"])

    def test_topic_words_in_the_title_are_a_full_match(self):
        self.reranker.scores.update({"Introductory Psychology": 0.2, "Film Studies": 0.1})
        courses = self.result(category="HUEL", about="psychology")["courses"]
        self.assertEqual([(course["code"], course["score"]["relevance"]) for course in courses], [("HSS F201", 1.0)])

    def test_equivalent_courses_share_one_entry_and_a_short_list_says_so(self):
        equivalent(self.courses["XX F411"], "XX F412")
        self.scores(**{"Natural Language Processing": 0.9, "Deep Learning": 0.9})
        result = self.result(category="DEL", about="linguistics", count=3)
        self.assertEqual([course["code"] for course in result["courses"]], ["XX F411"])
        self.assertIn("the same class is also offered as XX F412 (DEL); take only one", result["courses"][0]["note"])
        self.assertEqual(result["courses"][0]["also_offered_as"], [{"code": "XX F412", "title": "Deep Learning", "category": "DEL"}])
        self.assertEqual(result["shortfall"], "Only 1 of the 3 courses you asked for match 'linguistics' well and fit "
                                              "your timetable; no other course this semester does.")

    def test_same_class_shows_the_code_that_counts_best(self):
        opel = make_course("YY F411", "Natural Language Processing")
        offer(opel, [("lecture", "L1", {"T": [3], "Th": [3]})])
        equivalent(self.courses["XX F411"], "YY F411")
        self.embed({})
        self.scores(**{"Natural Language Processing": 0.9})
        result = self.result(about="linguistics")
        # YY F411 isn't linked to the programme: an OPEL here, the DEL wins even with the same relevance
        shown = {course["code"]: course for course in result["courses"]}
        self.assertNotIn("YY F411", shown)
        self.assertEqual(shown["XX F411"]["also_offered_as"], [{"code": "YY F411", "title": "Natural Language Processing", "category": "OPEL"}])

    def topic_vectors(self) -> None:
        """alpha: XX F411 and XX F412 (0.998, 0.990); beta: only XX F413 (0.979)."""
        self.embed({"alpha": [1, 0], "beta": [0, 1], "Natural Language Processing": [1, -0.1], "Deep Learning": [1, -0.2],
                    "Compiler Construction": [-0.3, 1]})

    def test_each_topic_gets_its_share(self):
        self.topic_vectors()
        result = self.result(category="DEL", about=["alpha", "beta"], count=2)
        # by rank alone both places would go to alpha; floor(2 / 2) = 1 place is kept for beta
        self.assertEqual([course["code"] for course in result["courses"]], ["XX F411", "XX F413"])
        result = self.result(category="DEL", about=["alpha"], count=2)
        self.assertEqual([course["code"] for course in result["courses"]], ["XX F411", "XX F412"])

    def test_default_count_is_per_category(self):
        self.reranker.scores.update({title: 0.9 for title in TITLES + ["Introductory Psychology", "Film Studies"]})
        result = self.result(about="anything")  # no category: DEL and HUEL both still needed
        categories = [course["category"] for course in result["courses"]]
        self.assertEqual((categories.count("DEL"), categories.count("HUEL")), (3, 2))  # every match, not 5 in total

    def test_04_penalties_reorder_but_never_remove(self):
        add_handout(self.courses["XX F411"], evaluation=[
            {"name": "Quizzes 1-4", "kind": "quiz", "weightage_percent": 20, "nature": "CB"},
            {"name": "Compre", "kind": "compre", "weightage_percent": 50, "nature": "CB"}], attendance_required=True,
            makeup_allowed=False)
        clear_facts_cache()
        self.student.avoid_eval_styles = ["many_quizzes", "closed_book", "strict_attendance", "heavy_compre", "no_makeup"]
        self.student.save()
        self.scores(**{"Natural Language Processing": 0.6, "Deep Learning": 0.55})
        courses = self.result(category="DEL", about="linguistics")["courses"]
        self.assertEqual([course["code"] for course in courses], ["XX F412", "XX F411"])
        self.assertEqual(courses[1]["score"]["final"], 0.35)  # below the cutoff after the penalty, still returned

    def test_05_only_top_candidates_reach_the_reranker_and_titles_always_do(self):
        course = self.courses["XX F413"]
        course.description = "Parsing and lexing. Code generation. Register allocation."
        course.save()
        self.embed({"q": [1, 0], "Natural Language Processing": [1, 0.1], "Deep Learning": [1, 0.2],
                    "Compiler Construction": [-1, 0], "Parsing and lexing.": [1, 0.05], "Code generation.": [0, 1],
                    "Register allocation.": [0, -1]})
        with patch.object(config, "RERANK_CANDIDATES", 2), patch.object(config, "RERANK_PIECES_PER_COURSE", 1):
            self.result(category="DEL", about="q")
        passages = [passage for _, passage in self.reranker.calls[-1]]
        # XX F413 ("Parsing" 0.99) and XX F411 (title 0.99) are the top 2; XX F412 (0.98) never reaches the reranker
        self.assertNotIn("Deep Learning", passages)
        self.assertIn("Parsing and lexing.", passages)
        self.assertIn("Compiler Construction", passages)  # the title rides along even with a low similarity
        self.assertNotIn("Code generation.", passages)    # only the best piece + the title

    def test_several_topics_are_matched_one_by_one(self):
        # "alpha" and "beta" point different ways; a course strongly about beta alone must count as a real match
        self.embed({"alpha": [1, 0], "beta": [0, 1], "Natural Language Processing": [0, 1],
                    "Deep Learning": [0.7, 0.7], "Compiler Construction": [-1, -1]})
        result = self.result(category="DEL", about=["alpha", "beta"])
        self.assertEqual([course["code"] for course in result["courses"]][:2], ["XX F411", "XX F412"])
        self.assertEqual(result["courses"][0]["score"]["relevance"], 1.0)

    def test_06_one_batched_rerank_call(self):
        unoffered = make_course("XX F499", "Language Security")  # reranked in the same call (not offered)
        link(self.programme, unoffered)
        self.embed({})
        self.scores(**{"Natural Language Processing": 0.9})
        calls = len(self.reranker.calls)
        self.result(about="language")
        self.assertEqual(len(self.reranker.calls), calls + 1)

    def test_09_keys_and_determinism(self):
        self.scores(**{"Natural Language Processing": 0.9})
        first = self.result(category="DEL", about="rocks")
        self.assertEqual(set(first), {"searched", "settings_used", "courses", "couldnt_verify", "excluded", "warnings",
                                      "shortfall"})
        self.assertEqual(set(first["courses"][0]["score"]), {"embedding", "relevance", "penalty", "final", "why", "topic"})
        self.assertEqual(self.result(category="DEL", about="rocks"), first)


class PrefixTests(SimpleTestCase):
    """Test 7: each model gets the query / passage prefix from its model card."""

    def encoded(self, model_name: str, kind: str) -> str:
        embedder = SentenceTransformerEmbedder(model_name)
        seen = []

        class Model:
            def encode(self, texts, **kwargs):
                seen.extend(texts)
                return [[1.0, 0.0]] * len(texts)
        embedder._model = Model()
        embedder.embed(["topic"], kind=kind)
        return seen[0]

    def test_07_prefixes(self):
        self.assertEqual(self.encoded("intfloat/e5-small-v2", "query"), "query: topic")
        self.assertEqual(self.encoded("intfloat/e5-small-v2", "passage"), "passage: topic")
        self.assertEqual(self.encoded("BAAI/bge-small-en-v1.5", "query"),
                         "Represent this sentence for searching relevant passages: topic")
        self.assertEqual(self.encoded("BAAI/bge-small-en-v1.5", "passage"), "topic")
        self.assertEqual(self.encoded("all-MiniLM-L6-v2", "query"), "topic")


class NeighbourTests(Catalog):
    """A topic no offered course matches directly: the courses closest in content to the catalogue's best match."""

    def setUp(self) -> None:
        super().setUp()
        link(self.programme, make_course("XX F499", "Video Production"))  # a DEL with no offering this semester
        patcher = patch.object(config, "NEIGHBOUR_ANCHOR_FLOOR", 0.3)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.embed({"Video Production": [1, 0], "Deep Learning": [0.9, 0.2], "Natural Language Processing": [0.2, 1],
                    "Compiler Construction": [-1, 0]})
        self.reranker.scores.update({title: 0.1 for title in TITLES + ["Introductory Psychology", "Film Studies",
                                                                       "Financial Markets", "Soil Science", "Data Structures"]})

    def test_closest_content_to_the_anchor_is_listed_after_a_warning(self):
        self.reranker.scores["Video Production"] = 0.4  # over the anchor floor, under the cutoff
        with patch.object(config, "NEIGHBOURS_PER_TOPIC", 2):
            result = self.result(category="DEL", about="video editing")
        self.assertEqual(self.codes(result), ["XX F412", "XX F411"])
        self.assertEqual(result["courses"][0]["score"]["similar_to"], "XX F499 Video Production")
        self.assertIn("no direct match for 'video editing'", result["courses"][0]["score"]["why"])
        self.assertIn("Nothing this semester matches 'video editing' directly; the courses listed for it are the closest "
                      "in content to the catalogue's best matches.", result["warnings"])

    def test_no_anchor_no_neighbours(self):
        self.reranker.scores["Video Production"] = 0.2  # "cooking": nothing in the catalogue is about it
        result = self.result(category="DEL", about="cooking")
        self.assertEqual(result["courses"], [])
        self.assertIn("Nothing this semester matches 'cooking' directly.", result["warnings"])


class TitleIndex:
    """Three courses, each a title piece and a topic piece."""
    texts = ["AI", "Search and planning", "AI in Civil Engineering", "Search and planning", "AI", "Search and planning"]
    kinds = ["title", "topic"] * 3


class PieceScores:
    def __init__(self, by_text: dict[str, float]) -> None:
        self.by_text = by_text

    def score(self, pairs):
        return np.array([self.by_text[text] for _, text in pairs])


class TitleTests(SimpleTestCase):
    def test_weak_title_is_ignored_not_averaged_in(self):
        item = Scored("CS F407", 0.9, [[0, 1]])
        with patch.object(config, "TITLE_STRONG", 0.8):
            rerank([[item]], ["artificial intelligence"], np.array([[1.0, 0.0]]), TitleIndex(),
                   PieceScores({"AI": 0.3, "Search and planning": 0.9}))
        # under TITLE_STRONG: relevance is the mean of the best pieces, with the title just one of them
        self.assertAlmostEqual(item.relevance, (0.9 + 0.3) / 2, places=5)
        self.assertEqual(TitleIndex.kinds[item.best_rows[0]], "topic")


class TitleWordTests(SimpleTestCase):
    def test_word_forms_match_but_different_words_sharing_a_prefix_dont(self):
        from recommender.ranking import title_contains
        self.assertTrue(title_contains("Politics and Society", "political"))
        self.assertTrue(title_contains("Statistical Inference", "statistics"))
        self.assertTrue(title_contains("Principles of Economics", "economy"))
        self.assertTrue(title_contains("Print and Audio-Visual Advertising", "advertisements"))
        self.assertTrue(title_contains("Machine Learning", "machines"))
        self.assertTrue(title_contains("Cinematic Art", "cinema"))
        self.assertFalse(title_contains("Artificial Intelligence", "art"))
        self.assertFalse(title_contains("Communication Skills", "communism"))  # a 7-letter prefix cut matched these
        self.assertFalse(title_contains("Community Development", "communism"))


class NamedCategoryTests(SimpleTestCase):
    def test_done_or_negated_categories_are_not_asks(self):
        from recommender.agent import named_categories
        self.assertEqual(named_categories("suggest DELs on ML, I've finished my HUELs"), {"DEL"})
        self.assertEqual(named_categories("no HUELs please, just OPELs"), {"OPEL"})
        self.assertEqual(named_categories("a DEL with no midsem"), {"DEL"})  # negation after the word is about the DEL
        self.assertEqual(named_categories("I don't want OPELs"), set())
