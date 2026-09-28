"""Pieces, build_embeddings and the piece index."""
from io import StringIO

import numpy as np
from django.core.management import call_command
from django.test import SimpleTestCase

from catalog.models import CoursePiece
from recommender.pieces import build_pieces, clean
from recommender.piece_index import get_piece_index

from .fixtures import RecommenderTestCase, add_handout, make_course


class CleanTests(SimpleTestCase):
    def test_labels(self):
        cases = {
            "1. Intro to ML": "Intro to ML", "1) Intro to ML": "Intro to ML", "(a) Graph theory": "Graph theory",
            "i. Set theory": "Set theory", "L1-3 Basic ideas": "Basic ideas", "Lec 4-6: Search trees": "Search trees",
            "Lecture 7 Heaps": "Heaps", "Week 2 Sorting": "Sorting", "Module 3: Hashing": "Hashing",
            "Unit II: Graphs": "Graphs", "• Bullet point": "Bullet point", "- Dash item": "Dash item",
            "* Star item": "Star item", "  lots   of \n space ": "lots of space", "7-9 Biotechnology of sewage": "Biotechnology of sewage",
            "2.5 GHz radios": "2.5 GHz radios", "3 phase motors": "3 phase motors", "L2 regularization": "L2 regularization",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(clean(raw), expected)


class BuildPiecesTests(RecommenderTestCase):
    def test_order_cleaning_and_dedupe(self):
        course = make_course("XX F101", "INTRODUCTION TO TESTING",
                             description="Testing basics. Unit tests and mocks.")
        add_handout(course, file="2_b.pdf", topics=["Mocks and fakes"], description="")
        add_handout(course, file="1_a.pdf", description="Why we test software. Testing basics.",
                    topics=["1. Unit tests and mocks", "Topics", "Fixtures", "Learning Outcomes:"],
                    learning_outcomes=["Write good tests"])
        self.assertEqual(build_pieces(course), [
            ("title", "Introduction to Testing", "timetable"),
            ("description", "Why we test software.", "1_a.pdf"),
            ("description", "Testing basics.", "1_a.pdf"),
            ("topic", "Unit tests and mocks", "1_a.pdf"),      # label stripped; "Topics" header and 1-word piece dropped
            ("outcome", "Write good tests", "1_a.pdf"),
            ("topic", "Mocks and fakes", "2_b.pdf"),
            # Bulletin sentences that repeat a handout piece (case-insensitively) are dropped
        ])

    def test_long_piece_split_into_sentences(self):
        long_sentence = "Word " + " ".join(["word"] * 70) + "."
        course = make_course("XX F102", "Thermodynamics", description=f"Short one here. {long_sentence}")
        add_handout(course, topics=["First part is here. More " + " ".join(["more"] * 65) + "."], description="")
        pieces = build_pieces(course)
        self.assertEqual(pieces[0], ("title", "Thermodynamics", "timetable"))  # one-word title still kept
        self.assertIn(("topic", "First part is here.", "XX_F102.pdf"), pieces)
        self.assertIn(("bulletin_description", long_sentence, "bulletin"), pieces)  # still > 60 words: kept whole


class BuildEmbeddingsTests(RecommenderTestCase):
    def test_command_and_index(self):
        course = make_course("XX F101", "Graph Theory", description="Paths and cycles. Colouring of maps.")
        make_course("XX F102", "Algebra")
        out = StringIO()
        call_command("build_embeddings", stdout=out)
        self.assertIn("2 courses, 4 pieces, 1 courses with only a title piece", out.getvalue())
        self.assertEqual(CoursePiece.objects.filter(course=course).count(), 3)

        index = get_piece_index()
        self.assertEqual(index.matrix.shape, (4, self.embedder.dimensions))
        self.assertEqual(index.course_codes, ["XX F101"] * 3 + ["XX F102"])
        self.assertEqual(index.rows_by_course, {"XX F101": [0, 1, 2], "XX F102": [3]})
        self.assertEqual(index.kinds[0], "title")
        self.assertTrue(np.allclose(np.linalg.norm(index.matrix, axis=1), 1))

        call_command("build_embeddings", stdout=StringIO())  # rerun replaces, and the index reloads
        self.assertEqual(CoursePiece.objects.count(), 4)
        self.assertIsNot(get_piece_index(), index)


class IngestHookTests(RecommenderTestCase):
    def test_ingest_builds_embeddings_unless_skipped(self):
        out = StringIO()
        call_command("ingest", stdout=out)
        self.assertIn("Embeddings (", out.getvalue())
        self.assertTrue(CoursePiece.objects.exists())
        out = StringIO()
        call_command("ingest", "--skip-embeddings", stdout=out)
        self.assertNotIn("Embeddings (", out.getvalue())
        self.assertFalse(CoursePiece.objects.exists())  # the catalog was reloaded; pieces cascade-deleted with it
