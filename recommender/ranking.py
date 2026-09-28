"""Stage C of get_eligible_courses (todo.md 6.4): retrieve with embeddings, then rerank with a cross-encoder.

C1 retrieval: a course's embedding score is its BEST piece similarity (not an average: one strongly matching
    title beats many vaguely related topics). The top `candidates` courses go on, each with its best
    RERANK_PIECES_PER_COURSE pieces plus its title.
C2 rerank: every (query, piece) pair of every candidate is scored by the cross-encoder in one batched call.
    relevance = the title's score if the title matches strongly, else the mean of the best RELEVANCE_TOP_PIECES
    pieces (a mean, so one passing mention can't carry a course).
    Without a reranker (EDA baseline), the embedding similarities of the same pieces are used instead.
Neighbours: when a topic finds too little, the courses whose content is closest to the catalogue's best matches
    for it (see neighbours()).
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
    rows_by_topic: list[list[int]]   # per topic: its best pieces by embedding + the title
    relevance: float | None = None   # the best topic's
    by_topic: list[float] = field(default_factory=list)   # relevance for each topic, in the order given
    best_rows: list[int] = field(default_factory=list)    # the piece that answered each topic


def retrieve(index: PieceIndex, codes: list[str], queries: np.ndarray, candidates: int) -> list[Scored]:
    """C1: for EACH topic, the `candidates` courses with the highest best-piece similarity to it (ties by code); the
    union, sorted by the best similarity to any topic. Each topic gets the course's RERANK_PIECES_PER_COURSE best
    pieces FOR THAT TOPIC, plus the title.
    why per topic: with "NLP" and "machine learning", the AI course's three ML lines would otherwise crowd out its
    one NLP line, and its NLP score would be computed without it. Courses without any piece are left out.
    why `candidates` per topic, not shared: "programming, algorithms" lost Object Oriented Programming (15th for
    "programming" alone) to algorithm courses. probe 2026-09-28: labelled queries unchanged (hit@5 0.80, MRR 0.82),
    +0.05 s; "ML, NLP, statistics" found 9 real matches instead of 4."""
    queries = np.atleast_2d(queries)
    scored, best_by_topic = [], []
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
        scored.append(Scored(code, float(sims.max()), by_topic))
        best_by_topic.append(sims.max(axis=0))
    kept = set()
    for topic in range(queries.shape[0]):
        order = sorted(range(len(scored)), key=lambda i: (-best_by_topic[i][topic], scored[i].code))
        kept.update(order[:candidates])
    return sorted((scored[i] for i in kept), key=lambda item: (-item.embedding, item.code))


# longest first; a closed list of word endings, so word forms of one root meet on the same base
SUFFIXES = ("ically", "ations", "ation", "ments", "ical", "ment", "ics", "ies", "ing", "al", "ic", "es", "ed", "s", "y")
MIN_BASE = 3  # "ethics" / "ethical" -> "eth"; shorter bases ("gas" -> "ga") are left whole
PREFIX_BASE = 6  # shorter bases ("art", "data") must match whole words, or "art" would match "artificial"


def base(word: str) -> str:
    """The word without its longest listed ending, then without a final "e": politics / political -> polit,
    advertisements / advertising -> advertis, course / courses -> cours; but communism / communication /
    community stay apart (a 7-letter prefix cut made all three "communi")."""
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_BASE:
            word = word[:-len(suffix)]
            break
    return word[:-1] if word.endswith("e") and len(word) > MIN_BASE + 1 else word


def words(text: str) -> set[str]:
    """Lowercase words, each reduced to its base (see base())."""
    return {base(word) for word in re.findall(r"[a-z0-9]+", text.lower())}


def title_contains(title: str, topic: str) -> bool:
    """Every word of the topic is in the title ("statistics" in "Statistics & Basic Econometrics").
    why: the cross-encoder scores a one-word topic against a short title badly (0.2 for that pair)."""
    wanted, have = words(topic), words(title)
    # a long base may also start a title word: cinema -> cinemat(ic), program -> programm(ing)
    return bool(wanted) and all(word in have or (len(word) >= PREFIX_BASE and any(other.startswith(word) for other in have))
                                for word in wanted)


def topic_score(index: PieceIndex, rows: list[int], scores: list[float], topic: str) -> tuple[float, int]:
    """(relevance, best piece position in `rows`) of one course for one topic."""
    titles = [i for i, row in enumerate(rows) if index.kinds[row] == "title"]
    for i in titles:
        if title_contains(index.texts[rows[i]], topic):
            scores[i] = 1.0
    title_score = max((scores[i] for i in titles), default=0.0)
    content = float(np.mean(sorted(scores, reverse=True)[:config.RELEVANCE_TOP_PIECES]))
    # why the title counts on its own: a course NAMED "Natural Language Processing" teaches it, even when its
    # topic lines ("N-gram models", "POS tagging") never repeat the phrase (the cross-encoder scores those ~0).
    # A weak title is ignored rather than averaged in, so it never drags a course down.
    if titles and title_score >= config.TITLE_STRONG and title_score >= content:
        return title_score, titles[0]
    return content, int(np.argmax(scores))


def rerank(groups: list[list[Scored]], topics: list[str], queries: np.ndarray, index: PieceIndex,
           reranker: Reranker | None) -> None:
    """C2: fill `by_topic`, `best_rows` and `relevance` (the best topic's) for every Scored in every group, with ONE
    reranker call for all pairs. Each topic is scored on its own pieces only (rows_by_topic)."""
    everyone = [item for group in groups for item in group]
    jobs = [(t, row) for item in everyone for t in range(len(topics)) for row in item.rows_by_topic[t]]
    if not jobs:
        return
    if reranker is not None:
        flat = np.asarray(reranker.score([(topics[t], index.texts[row]) for t, row in jobs]), dtype=float)
    else:  # no reranker: the same pieces, scored by their similarity
        vectors = np.atleast_2d(queries)
        flat = np.array([float(index.matrix[row] @ vectors[t]) for t, row in jobs])
    position = 0
    for item in everyone:
        item.by_topic, item.best_rows = [], []
        for t, topic in enumerate(topics):
            rows = item.rows_by_topic[t]
            score, best = topic_score(index, rows, [float(value) for value in flat[position:position + len(rows)]], topic)
            position += len(rows)
            item.by_topic.append(score)
            item.best_rows.append(rows[best])
        item.relevance = max(item.by_topic)


def closest(index: PieceIndex, code: str, profile: dict[str, np.ndarray]) -> tuple[str, float] | None:
    """(label, cosine) of the profile vector closest to the course's content; None without vectors on either side."""
    course = index.course_vectors.get(code)
    if course is None or not profile:
        return None
    return max(((label, float(course @ vector)) for label, vector in profile.items()), key=lambda pair: (pair[1], pair[0]))


def neighbours(index: PieceIndex, anchors: list[str], codes: list[str]) -> list[tuple[str, float]]:
    """`codes` by how close their content is to the anchors' (cosine of mean piece vectors), closest first.
    why: a topic the handouts never name ("video editing") still has a catalogue course about it (GS F343 Short
    Film and Video Production, not offered); the offered courses that teach the same material are its neighbours
    (GS F321 Mass Media Content & Design, GS F224 Print and Audio-Visual Advertising). No LLM involved."""
    centre = np.mean([index.course_vectors[code] for code in anchors], axis=0)
    found = [(code, float(index.course_vectors[code] @ centre)) for code in codes if code in index.course_vectors]
    return sorted(found, key=lambda pair: (-pair[1], pair[0]))
