"""Same-class detection from text (recommender/equivalents.py), on a tiny in-memory piece index."""
import numpy as np
from django.test import SimpleTestCase

from recommender.equivalents import overlaps
from recommender.piece_index import PieceIndex


def index(courses: dict[str, list[list[float]]]) -> PieceIndex:
    codes = [code for code, vectors in courses.items() for _ in vectors]
    matrix = np.array([vector for vectors in courses.values() for vector in vectors], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    built = PieceIndex("fake", matrix, codes, ["topic"] * len(codes), [""] * len(codes))
    for row, code in enumerate(codes):
        built.rows_by_course.setdefault(code, []).append(row)
    return built


class OverlapTests(SimpleTestCase):
    def test_smaller_course_contained_in_a_bigger_one(self):
        # BIG has 4 pieces; SMALL (a Bulletin entry) repeats 2 of them word for word; OTHER shares one line
        built = index({"BIG": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "SMALL": [[1, 0, 0, 0], [0, 1, 0, 0]],
                       "OTHER": [[1, 0, 0, 0], [0.2, 0.2, 0.2, -1]]})
        smaller = overlaps(built, {"BIG", "SMALL", "OTHER"}, 0.99, "smaller")
        both = overlaps(built, {"BIG", "SMALL", "OTHER"}, 0.99, "both")
        self.assertEqual(smaller[frozenset(("BIG", "SMALL"))], 1.0)   # every piece of the smaller course has a twin
        self.assertEqual(both[frozenset(("BIG", "SMALL"))], 0.5)      # only half of BIG is in SMALL
        self.assertEqual(smaller[frozenset(("BIG", "OTHER"))], 0.5)   # one shared line of two: not the same course
        self.assertEqual(overlaps(built, set(), 0.99, "smaller"), {})
