"""`python manage.py embedding_eda`: tune the retrieve-then-rerank ranking on real queries.

Grid: embedding model x reranker (or none) x RERANK_CANDIDATES, using the same code as the
app (recommender/ranking.py). Pieces are embedded in memory per model; stored CoursePiece rows are not touched.
Courses ranked = every offered course a pre-2026 student can take (category ignored: this tests matching only).

Per configuration: hit@5, MRR, recall@candidates, median latency per query, and the relevance cutoff that best
separates expected courses from other candidates (max F1), checked on the no_match queries.
Writes docs/eda/embedding_eda.md and docs/eda/pieces_per_course.png.
"""
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand

from catalog.models import Course, Offering
from recommender import config
from recommender.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
from recommender.piece_index import PieceIndex
from recommender.pieces import build_pieces
from recommender.ranking import rerank, retrieve

QUERIES = Path(__file__).resolve().parents[2] / "eda" / "queries.json"
OUT_DIR = settings.BASE_DIR / "docs" / "eda"
TOP_K = 5
MAX_LATENCY = 1.5  # seconds, median per query
HISTOGRAM_BUCKETS = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 10_000)]
CUTOFF_BINS = [round(0.1 * step, 1) for step in range(11)]


@dataclass
class Config:
    embedder: str
    reranker: str | None
    candidates: int
    hit: float = 0.0
    mrr: float = 0.0
    recall: float = 0.0
    latency: float = 0.0
    cutoff: float = 0.0
    precision: float = 0.0
    cutoff_recall: float = 0.0
    no_match_ok: int = 0
    rankings: dict[str, list[tuple[str, float]]] = field(default_factory=dict)  # query -> [(code, relevance)]
    labelled: list[tuple[float, bool]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.embedder} + {self.reranker or 'none'}, K={self.candidates}"


def pool() -> list[Course]:
    """Offered courses a pre-2026 student could take (2026-only duplicates like CS U429 left out)."""
    tag = Offering.objects.order_by("pk").values_list("semester_tag", flat=True).first()
    return list(Course.objects.filter(offerings__semester_tag=tag, offerings__sections__cancelled=False,
                                      only_2026_batch=False).distinct().prefetch_related("handouts").order_by("code"))


def build_index(model: str, pieces: list[tuple[str, str, str]]) -> PieceIndex:
    """Embed the pieces with `model` (passage prefix) into an in-memory PieceIndex."""
    matrix = SentenceTransformerEmbedder(model).embed([text for *_, text in pieces], kind="passage")
    return PieceIndex(model, matrix, [code for code, *_ in pieces], [kind for _, kind, _ in pieces], [text for *_, text in pieces])


def best_cutoff(labelled: list[tuple[float, bool]]) -> tuple[float, float, float]:
    """(cutoff, precision, recall) maximising F1 of "relevance >= cutoff" as a predictor of "is expected"."""
    positives = sum(label for _, label in labelled)
    best = (0.0, 0.0, 0.0, -1.0)
    for threshold in sorted({round(value, 3) for value, _ in labelled}):
        predicted = [label for value, label in labelled if value >= threshold]
        hits = sum(predicted)
        precision = hits / len(predicted) if predicted else 0.0
        recall = hits / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best[3]:
            best = (threshold, precision, recall, f1)
    return best[:3]


class Command(BaseCommand):
    help = "Tune embedding model, reranker, RERANK_CANDIDATES and RELEVANCE_CUTOFF."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--models", nargs="+", default=["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2"])
        parser.add_argument("--rerankers", nargs="+", default=["none", "cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"])
        parser.add_argument("--candidates", nargs="+", type=int, default=[30, 50])

    def handle(self, *args, **options) -> None:
        queries = json.loads(QUERIES.read_text())
        courses = pool()
        pieces = [(course.code, kind, text) for course in courses for kind, text, _ in build_pieces(course)]
        codes = [course.code for course in courses]
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        rerankers = {name: None if name == "none" else CrossEncoderReranker(name) for name in options["rerankers"]}
        results: list[Config] = []
        for model in options["models"]:
            self.stdout.write(f"embedding {len(pieces)} pieces with {model} ...")
            index = build_index(model, pieces)
            embedder = SentenceTransformerEmbedder(model)
            embedder.embed(["warm up"], kind="query")
            for reranker_name, reranker in rerankers.items():
                if reranker is not None:
                    reranker.score([("warm up", "warm up")])  # model load isn't query latency
                results += self.run_grid(model, reranker_name, reranker, embedder, index, codes, queries, options)
        self.score(results, queries)
        self.write_report(results, queries, courses, pieces)

    def run_grid(self, model, reranker_name, reranker, embedder, index, codes, queries, options) -> list[Config]:
        """All candidate counts for one embedder + reranker."""
        results = []
        for k in options["candidates"]:
            found, latencies = Config(model, None if reranker_name == "none" else reranker_name, k), []
            for query in queries:
                started = time.monotonic()
                topics = [topic.strip() for topic in query["query"].split(",")]  # queries.json: comma-separated topics
                vectors = embedder.embed(topics, kind="query")
                scored = retrieve(index, codes, vectors, k)
                rerank([scored], topics, vectors, index, reranker)
                latencies.append(time.monotonic() - started)
                found.rankings[query["query"]] = sorted(((item.code, item.relevance) for item in scored),
                                                        key=lambda pair: (-pair[1], pair[0]))
            found.latency = statistics.median(latencies)
            self.stdout.write(f"  {model} + {reranker_name}, K={k}: median latency {found.latency:.2f}s")
            results.append(found)
        return results

    def score(self, results: list[Config], queries: list[dict]) -> None:
        """hit@5, MRR, recall@candidates over the queries with expected codes; the cutoff and the no_match check."""
        judged = [query for query in queries if query["expected"]]
        no_match = [query for query in queries if query.get("no_match")]
        for found in results:
            hits, ranks, recalls = [], [], []
            for query in judged:
                ranking = found.rankings[query["query"]]
                order = [code for code, _ in ranking]
                expected = set(query["expected"])
                hits.append(len(expected & set(order[:TOP_K])) / len(expected))
                first = next((position for position, code in enumerate(order, 1) if code in expected), None)
                ranks.append(1 / first if first else 0.0)
                recalls.append(len(expected & set(order)) / len(expected))
                found.labelled += [(value, code in expected) for code, value in ranking]
            found.hit, found.mrr, found.recall = map(lambda values: float(np.mean(values)), (hits, ranks, recalls))
            found.cutoff, found.precision, found.cutoff_recall = best_cutoff(found.labelled)
            found.no_match_ok = sum(all(value < found.cutoff for _, value in found.rankings[query["query"]]) for query in no_match)

    # ------------------------------------------------------------ report
    def write_report(self, results: list[Config], queries: list[dict], courses: list[Course], pieces: list[tuple]) -> None:
        fast = [found for found in results if found.latency <= MAX_LATENCY]
        chosen = sorted(fast, key=lambda f: (-f.hit, -f.mrr, -f.no_match_ok, f.latency))[0]
        titles = {course.code: course.title for course in courses}
        no_match = sum(bool(query.get("no_match")) for query in queries)
        lines = ["# Embedding + reranker EDA", "",
                 f"Generated by `python manage.py embedding_eda` over {len(courses)} offered courses a pre-2026 student can take "
                 f"({len(pieces)} pieces). Queries: `recommender/eda/queries.json` ({len(queries) - no_match} with expected codes, "
                 f"{no_match} that should match nothing). Category ignored: this tests matching only.", "",
                 "## Chosen configuration", "",
                 f"**{chosen.name}**, relevance cutoff **{chosen.cutoff:.2f}**.", "",
                 f"Rule: highest hit@5, then MRR, then no-match correctness, then lower latency; configurations with a median "
                 f"latency above {MAX_LATENCY} s per query dropped ({len(results) - len(fast)} dropped).", "",
                 f"hit@5 {chosen.hit:.2f} · MRR {chosen.mrr:.2f} · recall@candidates {chosen.recall:.2f} · median latency "
                 f"{chosen.latency:.2f} s · at the cutoff: precision {chosen.precision:.2f}, recall {chosen.cutoff_recall:.2f}, "
                 f"no-match queries with no real match {chosen.no_match_ok}/{no_match}.", "",
                 "## All configurations", "",
                 "| embedder | reranker | K | hit@5 | MRR | recall@K | latency s | cutoff | precision | recall | no-match ok |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for found in sorted(results, key=lambda f: (-f.hit, -f.mrr, -f.no_match_ok, f.latency)):
            lines.append(f"| {found.embedder} | {found.reranker or 'none'} | {found.candidates} | {found.hit:.2f} | "
                         f"{found.mrr:.2f} | {found.recall:.2f} | {found.latency:.2f} | {found.cutoff:.2f} | {found.precision:.2f} | "
                         f"{found.cutoff_recall:.2f} | {found.no_match_ok}/{no_match} |")
        lines += ["", "## Cutoff data (chosen configuration)", "",
                  "Candidates per relevance bin: expected courses vs everything else (queries with expected codes).", "",
                  "| relevance | expected | other |", "|---|---|---|"]
        for low, high in zip(CUTOFF_BINS, CUTOFF_BINS[1:]):
            inside = [label for value, label in chosen.labelled if low <= value < high or (high == 1.0 and value == 1.0)]
            lines.append(f"| {low:.1f}-{high:.1f} | {sum(inside)} | {len(inside) - sum(inside)} |")
        lines += ["", "## Top 5 per query (chosen configuration)", ""]
        for query in queries:
            ranking = chosen.rankings[query["query"]]
            real = sum(value >= chosen.cutoff for _, value in ranking)
            lines += [f"### {query['query']}", "",
                      f"Real matches (relevance >= cutoff): {real}. Expected: {', '.join(query['expected']) or 'none'}.", "",
                      "| # | code | title | relevance | expected |", "|---|---|---|---|---|"]
            lines += [f"| {n} | {code} | {titles.get(code, '')} | {value:.2f} | {'yes' if code in query['expected'] else ''} |"
                      for n, (code, value) in enumerate(ranking[:TOP_K], 1)]
            lines.append("")
        lines += self.piece_stats(pieces, courses)
        (OUT_DIR / "embedding_eda.md").write_text("\n".join(lines) + "\n")
        self.stdout.write("\n".join(lines[:12]) + f"\n\nReport: {OUT_DIR / 'embedding_eda.md'}")

    def piece_stats(self, pieces: list[tuple], courses: list[Course]) -> list[str]:
        per_course = Counter(code for code, *_ in pieces)
        counts = [per_course.get(course.code, 0) for course in courses]
        self.histogram_png(counts)
        lines = ["## Pieces", "", f"{len(pieces)} pieces for {len(counts)} courses; only a title piece: {sum(c == 1 for c in counts)}.",
                 "", "```"]
        widest = max(1, max(sum(low <= c <= high for c in counts) for low, high in HISTOGRAM_BUCKETS))
        for low, high in HISTOGRAM_BUCKETS:
            number = sum(low <= c <= high for c in counts)
            label = f"{low}" if low == high else f"{low}+" if high >= 10_000 else f"{low}-{high}"
            lines.append(f"{label:>6} | {'#' * round(40 * number / widest):40} {number}")
        lines += ["```", "", "![pieces per course](pieces_per_course.png)", "", "| kind | pieces |", "|---|---|"]
        lines += [f"| {kind} | {count} |" for kind, count in Counter(kind for _, kind, _ in pieces).most_common()]
        return lines

    def histogram_png(self, counts: list[int]) -> None:
        import matplotlib
        matplotlib.use("Agg")  # why: no window, just the file
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(figsize=(7, 3.5))
        axes.hist(counts, bins=range(0, max(counts) + 2), color="#2f3fc4")
        axes.set_xlabel("pieces per course")
        axes.set_ylabel("courses")
        figure.tight_layout()
        figure.savefig(OUT_DIR / "pieces_per_course.png", dpi=120)
        plt.close(figure)
