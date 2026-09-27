"""Stage C of get_eligible_courses (todo.md 6.4): retrieve with embeddings, then rerank with a cross-encoder.

C1 retrieval: a course's embedding score is its BEST piece similarity (not an average: one strongly matching
    title beats many vaguely related topics). The top `candidates` courses go on, each with its best
    RERANK_PIECES_PER_COURSE pieces plus its title.
C2 rerank: every (query, piece) pair of every candidate is scored by the cross-encoder in one batched call.
    content = w * best piece + (1 - w) * mean of the best RELEVANCE_TOP_PIECES pieces (w = 0: the plain mean, so one
    passing mention can't carry a course).
    relevance = max(content, title score if the title matches strongly) * domain fit.
    Without a reranker (EDA baseline), the embedding similarities of the same pieces are used instead.
Domain fit: how close the course's DEPARTMENT is to the topic (department vectors, piece_index.py), min-max scaled
    over departments, so "Applications of AI in Civil Engineering" ranks under CS's AI course for "AI".
The query is a LIST of topics (the LLM separates them: ["machine learning", "natural language processing"]); each
is scored on its own and a course counts by its best match against ANY topic. why: one blended embedding of
"AI, ML, DL, NLP" half-matches every course and fully matches none.
Also used by the EDA command, so it measures exactly what the app does.
"""
import re

from dataclasses import dataclass, field

import numpy as np

from . import config
from .embeddings import Reranker
from .piece_index import PieceIndex


@dataclass
class Scored:
    """One course's scores for one query."""

    code: str
    embedding: float
    rows: list[int]                  # the pieces sent to the reranker (best by embedding, plus the title)
    relevance: float | None = None
    best_row: int | None = None      # the piece the reranker liked best
    best_topic: str | None = None    # the student's topic that matched best_row
    domain: float = 1.0              # the multiplier from the department's fit with the topic (1 = no change)
    piece_scores: list[float] = field(default_factory=list)


def piece_sims(index: PieceIndex, rows: list[int], queries: np.ndarray) -> np.ndarray:
    """Each piece's best similarity over the topics: queries is (topics, d)."""
    return (index.matrix[rows] @ np.atleast_2d(queries).T).max(axis=1)


def retrieve(index: PieceIndex, codes: list[str], queries: np.ndarray, candidates: int) -> list[Scored]:
    """C1: the `candidates` courses with the highest best-piece similarity (ties by code), with their pieces.
    Courses without any piece can't be scored and are left out."""
    scored = []
    for code in codes:
        rows = index.rows_by_course.get(code)
        if not rows:
            continue
        sims = piece_sims(index, rows, queries)
        order = np.argsort(-sims, kind="stable")
        chosen = [rows[i] for i in order[:config.RERANK_PIECES_PER_COURSE]]
        chosen += [row for row in rows if index.kinds[row] == "title" and row not in chosen]
        scored.append(Scored(code, float(sims[order[0]]), chosen))
    return sorted(scored, key=lambda item: (-item.embedding, item.code))[:candidates]


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def title_contains(title: str, topic: str) -> bool:
    """Every word of the topic is in the title ("statistics" in "Statistics & Basic Econometrics").
    why: the cross-encoder scores a one-word topic against a short title badly (0.2 for that pair)."""
    wanted = words(topic)
    return bool(wanted) and wanted <= words(title)


def relevance(piece_scores: list[float], weight: float) -> float:
    """w * best + (1 - w) * mean of the best few."""
    ordered = sorted(piece_scores, reverse=True)
    return weight * ordered[0] + (1 - weight) * float(np.mean(ordered[:config.RELEVANCE_TOP_PIECES]))


def domain_fits(index: PieceIndex, queries: np.ndarray) -> np.ndarray | None:
    """(n_departments, topics): each department's similarity to each topic, scaled 0-1 over departments per topic.
    None when there's nothing to compare (no department vectors, or a single department)."""
    if index.department_matrix is None or len(index.departments) < 2:
        return None
    sims = index.department_matrix @ np.atleast_2d(queries).T
    low, high = sims.min(axis=0), sims.max(axis=0)
    return (sims - low) / np.where(high > low, high - low, 1.0)


def domain_multiplier(index: PieceIndex, fits: np.ndarray | None, code: str, topic: int) -> float:
    """1 - DOMAIN_WEIGHT * (1 - fit): the worst-fitting department loses DOMAIN_WEIGHT of its score, never more.
    Interdisciplinary prefixes (BITS) aren't a subject, so they get no penalty."""
    department = code.split()[0]
    if fits is None or department in config.DOMAIN_NEUTRAL_DEPARTMENTS or department not in index.departments:
        return 1.0
    return 1 - config.DOMAIN_WEIGHT * (1 - float(fits[index.departments[department], topic]))


def rerank(groups: list[list[Scored]], topics: list[str], queries: np.ndarray, index: PieceIndex,
           reranker: Reranker | None, weight: float) -> None:
    """C2: fill `relevance` / `best_row` for every Scored in every group, with ONE reranker call for all pairs
    (every topic x every piece; a piece keeps its best topic's score)."""
    everyone = [item for group in groups for item in group]
    rows = [row for item in everyone for row in item.rows]
    if reranker is not None:
        pairs = [(topic, index.texts[row]) for row in rows for topic in topics]
        flat = reranker.score(pairs) if pairs else np.zeros(0)
        per_topic = flat.reshape(len(rows), len(topics))
    else:  # no reranker: the same pieces, scored by their similarity
        per_topic = index.matrix[rows] @ np.atleast_2d(queries).T
    if not rows:
        return
    scores, topic_of_row = per_topic.max(axis=1), per_topic.argmax(axis=1)
    fits = domain_fits(index, queries)
    position = 0
    for item in everyone:
        item.piece_scores = [float(value) for value in scores[position:position + len(item.rows)]]
        position += len(item.rows)
        titles = [i for i, row in enumerate(item.rows) if index.kinds[row] == "title"]
        for i in titles:
            if any(title_contains(index.texts[item.rows[i]], topic) for topic in topics):
                item.piece_scores[i] = 1.0
        title_score = max((item.piece_scores[i] for i in titles), default=0.0)
        content = relevance(item.piece_scores, weight)
        # why the title counts on its own: a course NAMED "Natural Language Processing" teaches it, even when its
        # topic lines ("N-gram models", "POS tagging") never repeat the phrase (the cross-encoder scores those ~0).
        # A weak title is ignored rather than averaged in, so it never drags a course down.
        uses_title = bool(titles) and title_score >= config.TITLE_STRONG and title_score >= content
        best = titles[0] if uses_title else int(np.argmax(item.piece_scores))
        topic = int(topic_of_row[position - len(item.rows) + best])
        item.domain = domain_multiplier(index, fits, item.code, topic)
        item.relevance = (title_score if uses_title else content) * item.domain
        item.best_row = item.rows[best]
        item.best_topic = topics[topic]
