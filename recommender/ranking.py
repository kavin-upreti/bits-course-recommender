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
    rows: list[int]                  # every piece sent to the reranker (the union over topics)
    rows_by_topic: list[list[int]] = field(default_factory=list)  # per topic: its best pieces by embedding + the title
    relevance: float | None = None
    best_row: int | None = None      # the piece the reranker liked best
    best_topic: str | None = None    # the student's topic that matched best_row
    domain: float = 1.0              # the multiplier from the department's fit with the topic (1 = no change)
    piece_scores: list[float] = field(default_factory=list)
    by_topic: list[float] = field(default_factory=list)   # relevance for each topic, in the order given
    best_rows: list[int] = field(default_factory=list)    # the piece that answered each topic
    domains: list[float] = field(default_factory=list)    # the department-fit multiplier for each topic


def retrieve(index: PieceIndex, codes: list[str], queries: np.ndarray, candidates: int,
             own: int | None = None) -> list[Scored]:
    """C1: the `candidates` courses with the highest best-piece similarity to any topic (ties by code), plus the
    `candidates` best on the first `own` topics (the student's), so related topics can't crowd those out. Each topic
    gets the course's RERANK_PIECES_PER_COURSE best pieces FOR THAT TOPIC, plus the title.
    why per topic: with "NLP" and "machine learning", the AI course's three ML lines would otherwise crowd out its
    one NLP line, and its NLP score would be computed without it. Courses without any piece are left out."""
    queries = np.atleast_2d(queries)
    scored, own_sim = [], {}
    for code in codes:
        rows = index.rows_by_course.get(code)
        if not rows:
            continue
        sims = index.matrix[rows] @ queries.T  # (pieces, topics)
        titles = [row for row in rows if index.kinds[row] == "title"]
        by_topic = []
        for topic in range(queries.shape[0]):
            best = [rows[i] for i in np.argsort(-sims[:, topic], kind="stable")[:config.RERANK_PIECES_PER_COURSE]]
            by_topic.append(best + [row for row in titles if row not in best])
        union = list(dict.fromkeys(row for chosen in by_topic for row in chosen))
        scored.append(Scored(code, float(sims.max()), union, by_topic))
        own_sim[code] = float(sims[:, :own].max()) if own else 0.0
    # ponytail: up to 2 x `candidates` reach the reranker when related topics are given; split the budget if that's slow
    shortlist = sorted(scored, key=lambda item: (-item.embedding, item.code))[:candidates]
    if own:
        listed = {item.code for item in shortlist}
        shortlist += [item for item in sorted(scored, key=lambda item: (-own_sim[item.code], item.code))[:candidates]
                      if item.code not in listed]
    return sorted(shortlist, key=lambda item: (-item.embedding, item.code))


# longest first; a closed list of word endings, so word forms of one root meet on the same base
SUFFIXES = ("ically", "ations", "ation", "ical", "ics", "ies", "ing", "al", "ic", "es", "ed", "s", "y")
MIN_BASE = 3  # "ethics" / "ethical" -> "eth"; shorter bases ("gas" -> "ga") are left whole


def base(word: str) -> str:
    """The word without its longest listed ending: politics / political -> polit, economy / economics -> econom,
    but communism / communication / community stay apart (a 7-letter prefix cut made all three "communi")."""
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_BASE:
            return word[:-len(suffix)]
    return word


def words(text: str) -> set[str]:
    """Lowercase words, each reduced to its base (see base())."""
    return {base(word) for word in re.findall(r"[a-z0-9]+", text.lower())}


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


def topic_score(index: PieceIndex, rows: list[int], scores: list[float], topic: str, weight: float,
                fit: float) -> tuple[float, int, float]:
    """(relevance, best piece position in `rows`, domain multiplier) of one course for one topic."""
    titles = [i for i, row in enumerate(rows) if index.kinds[row] == "title"]
    for i in titles:
        if title_contains(index.texts[rows[i]], topic):
            scores[i] = 1.0
    title_score = max((scores[i] for i in titles), default=0.0)
    content = relevance(scores, weight)
    # why the title counts on its own: a course NAMED "Natural Language Processing" teaches it, even when its
    # topic lines ("N-gram models", "POS tagging") never repeat the phrase (the cross-encoder scores those ~0).
    # A weak title is ignored rather than averaged in, so it never drags a course down.
    uses_title = bool(titles) and title_score >= config.TITLE_STRONG and title_score >= content
    best = titles[0] if uses_title else int(np.argmax(scores))
    return (title_score if uses_title else content) * fit, best, fit


def rerank(groups: list[list[Scored]], topics: list[str], queries: np.ndarray, index: PieceIndex,
           reranker: Reranker | None, weight: float) -> None:
    """C2: fill `by_topic`, `relevance` (the best topic's), `best_row` / `best_topic` for every Scored in every group,
    with ONE reranker call for all pairs. Each topic is scored on its own pieces only (rows_by_topic), so the pairs
    grow with topics x pieces, not topics x all pieces of every topic."""
    everyone = [item for group in groups for item in group]
    for item in everyone:
        if not item.rows_by_topic:  # built by hand (tests): every topic gets every piece
            item.rows_by_topic = [item.rows] * len(topics)
    jobs = [(t, row) for item in everyone for t in range(len(topics)) for row in item.rows_by_topic[t]]
    if not jobs:
        return
    if reranker is not None:
        flat = np.asarray(reranker.score([(topics[t], index.texts[row]) for t, row in jobs]), dtype=float)
    else:  # no reranker: the same pieces, scored by their similarity
        vectors = np.atleast_2d(queries)
        flat = np.array([float(index.matrix[row] @ vectors[t]) for t, row in jobs])
    fits = domain_fits(index, queries)
    position = 0
    for item in everyone:
        results = []
        for t, topic in enumerate(topics):
            rows = item.rows_by_topic[t]
            scores = [float(value) for value in flat[position:position + len(rows)]]
            position += len(rows)
            score, best, fit = topic_score(index, rows, scores, topic, weight, domain_multiplier(index, fits, item.code, t))
            results.append((score, rows[best], fit, scores))
        item.by_topic = [score for score, *_ in results]
        item.best_rows = [row for _, row, _, _ in results]
        item.domains = [fit for _, _, fit, _ in results]
        top = int(np.argmax(item.by_topic))
        item.relevance, item.domain = item.by_topic[top], item.domains[top]
        item.best_row, item.best_topic = item.best_rows[top], topics[top]
        item.piece_scores = results[top][3]
