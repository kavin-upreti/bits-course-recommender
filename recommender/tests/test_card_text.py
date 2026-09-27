"""Quiz display, reply bullet stripping and the matched topic (no DB needed)."""
from django.test import SimpleTestCase

import numpy as np

from recommender.agent import without_card_bullets
from recommender.cards import quiz_text
from recommender.handout_facts import CourseFacts
from recommender.ranking import Scored, rerank


class FakeIndex:
    texts = ["Deep learning", "Marketing mix"]
    kinds = ["topic", "topic"]
    department_matrix = None
    departments: dict = {}


class TopicReranker:
    def score(self, pairs):
        return np.array([1.0 if (topic, text) == ("economics", "Marketing mix") else 0.1 for topic, text in pairs])


class CardTextTests(SimpleTestCase):
    def test_quiz_text(self):
        facts = lambda **kw: CourseFacts(code="X", evaluation_parsed=True, **kw)  # noqa: E731
        self.assertEqual(quiz_text(facts(quiz_count=None, quiz_percent=10.0)), "Yes (10%)")  # "Quizes" without a count
        self.assertEqual(quiz_text(facts(quiz_count=2, quiz_percent=20.0)), "2 (20%)")
        self.assertEqual(quiz_text(facts(quiz_count=0, quiz_percent=0.0)), "None")
        self.assertIsNone(quiz_text(facts()))

    def test_without_card_bullets(self):
        reply = "Here you go:\n\n- **CS F407 Artificial Intelligence** (DEL): AI.\n- ME F321 note\n\nOne note about CS F425."
        self.assertEqual(without_card_bullets(reply, [{"code": "CS F407"}]),
                         "Here you go:\n\n- ME F321 note\n\nOne note about CS F425.")

    def test_best_topic(self):
        item = Scored("X", 0.5, [0, 1])
        rerank([[item]], ["machine learning", "economics"], np.zeros((2, 2)), FakeIndex(), TopicReranker(), 1.0)
        self.assertEqual((item.best_row, item.best_topic), (1, "economics"))
