"""Courses that are the same class under two codes (EEE F434 = ECE F434, CS F215 = ECE F215), found from their text.

The piece matrix is dotted with itself; two pieces of different courses at >= EQUIVALENT_TWIN_SIM are "twins".
overlap(A, B) = share of the SMALLER course's pieces with a twin in the other (EQUIVALENT_MEASURE "smaller"), or the
lower of both shares ("both"). why "smaller" exists: one code often has a 40-piece handout and its twin only a 3-piece
Bulletin entry, so "both" can never be high for them. Pairs at >= EQUIVALENT_OVERLAP
become CourseEquivalent rows (source "content"), next to the timetable's and the Bulletin's own "Equivalent:" lists.
Run by build_embeddings (so after every ingest); thresholds from `python manage.py equivalence_eda`.
"""
from collections import defaultdict

import numpy as np

from catalog.models import Course, CourseEquivalent

from . import config
from .piece_index import PieceIndex

BLOCK_ROWS = 2048  # rows of the piece matrix per matmul block: keeps the similarity block at ~2048 x n floats


def twin_counts(index: PieceIndex, codes: set[str], twin_sim: float) -> dict[tuple[str, str], int]:
    """(A, B) -> how many of A's pieces have a twin in B, for courses in `codes`."""
    if not codes:
        return {}
    rows = np.array([row for code in sorted(codes) for row in index.rows_by_course[code]], dtype=int)
    owner = np.array([index.course_codes[row] for row in rows])
    matrix = index.matrix[rows]
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for start in range(0, len(rows), BLOCK_ROWS):
        sims = matrix[start:start + BLOCK_ROWS] @ matrix.T
        left, right = np.nonzero(sims >= twin_sim)
        left += start
        other = owner[left] != owner[right]
        # a piece counts once per other course, however many twins it has there
        for piece, course in {(int(i), owner[j]) for i, j in zip(left[other], right[other])}:
            counts[(owner[piece], course)] += 1
    return counts


def overlap(a_share: float, b_share: float, a_size: int, b_size: int, measure: str) -> float:
    if measure == "both":
        return min(a_share, b_share)
    return a_share if (a_size, a_share) < (b_size, b_share) else b_share  # the smaller course's share


def overlaps(index: PieceIndex, codes: set[str], twin_sim: float, measure: str) -> dict[frozenset, float]:
    """Every pair with at least one twin in each direction -> overlap (0-1)."""
    counts = twin_counts(index, codes, twin_sim)
    sizes = {code: len(index.rows_by_course[code]) for code in codes}
    return {frozenset((a, b)): overlap(count / sizes[a], counts[(b, a)] / sizes[b], sizes[a], sizes[b], measure)
            for (a, b), count in counts.items() if a < b and counts.get((b, a))}


def comparable_codes(index: PieceIndex) -> set[str]:
    """Courses worth comparing: project courses share boilerplate text ("Study Project") without being one class,
    and a title alone ("Mechanics of Solids" in CE / ME) says nothing about the content."""
    projects = set(Course.objects.filter(is_project_course=True).values_list("code", flat=True))
    return {code for code, rows in index.rows_by_course.items() if code not in projects and len(rows) > 1}


def lecture_times(codes: set[str]) -> dict[str, set[str]]:
    """Code -> its lecture slots this semester ("M3", "W3"), for offered courses with timed lectures."""
    slots: dict[str, set[str]] = defaultdict(set)
    for course in Course.objects.filter(code__in=codes).prefetch_related("offerings__sections"):
        for offering in course.offerings.all():
            for section in offering.sections.all():
                if section.type == "lecture" and not section.cancelled:
                    slots[course.code] |= {f"{day}{period}" for day, periods in section.timings.items() for period in periods}
    return {code: found for code, found in slots.items() if found}


def separate_classes(pairs: set[frozenset]) -> set[frozenset]:
    """Pairs the timetable shows as two classes: both have lectures this semester and never at the same time.
    why: BIO F212 Microbiology and BIO G523 Advanced Microbiology share a professor and most handout text, yet meet
    at different hours; text alone would call them one course."""
    slots = lecture_times({code for pair in pairs for code in pair})
    return {pair for pair in pairs if all(code in slots for code in pair) and not set.intersection(*(slots[code] for code in pair))}


def detect(index: PieceIndex) -> list[tuple[str, str, float]]:
    """(code, code, overlap) for every pair at or above the thresholds in config, minus separate classes."""
    found = overlaps(index, comparable_codes(index), config.EQUIVALENT_TWIN_SIM, config.EQUIVALENT_MEASURE)
    above = {pair for pair, value in found.items() if value >= config.EQUIVALENT_OVERLAP}
    return sorted((*sorted(pair), found[pair]) for pair in above - separate_classes(above))


def store(pairs: list[tuple[str, str, float]]) -> int:
    """Replace the content-detected rows; pairs the timetable / Bulletin already list are left as they are."""
    CourseEquivalent.objects.filter(source="content").delete()
    courses = {course.code: course for course in Course.objects.filter(code__in={a for a, _, _ in pairs})}
    created = CourseEquivalent.objects.bulk_create(
        [CourseEquivalent(course=courses[a], equivalent_code=b, source="content") for a, b, _ in pairs],
        ignore_conflicts=True)
    return len(created)
