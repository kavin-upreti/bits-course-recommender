"""`python manage.py equivalence_eda`: pick EQUIVALENT_TWIN_SIM and EQUIVALENT_OVERLAP (recommender/equivalents.py).

Labels: SAME = pairs the timetable or the Bulletin list as equivalent, plus timetable cross-listings (same title and the
same section list: rooms, times, instructors). DIFFERENT = pairs with different titles that aren't listed.
Same-title pairs that aren't listed are UNLABELLED (they're exactly the unknowns: CE F211 and ME F211 "Mechanics of
Solids" are separate classes, BIO F243 and BIOT F243 "Genetics" are one) and are left out of the scores, then listed.
Pairs the timetable shows as separate classes (lectures never at the same time) are dropped, as in the app.
Grid: overlap measure x twin similarity x threshold; the best F1 wins, ties to the HIGHER threshold (a false match hides
a course as "already done", which is worse than showing a duplicate).
Writes docs/eda/equivalence_eda.md. Uses the stored pieces (run build_embeddings first).
"""
import re
from collections import defaultdict
from itertools import combinations

from django.conf import settings
from django.core.management.base import BaseCommand

from catalog.models import Course, CourseEquivalent
from recommender import config
from recommender.equivalents import comparable_codes, overlaps, separate_classes
from recommender.piece_index import get_piece_index

TWIN_SIMS = [0.95, 0.97, 0.99]
MEASURES = ["smaller", "both"]
THRESHOLDS = [round(0.05 * step, 2) for step in range(1, 21)]
OUT = settings.BASE_DIR / "docs" / "eda" / "equivalence_eda.md"
SHOWN = 20


def normal_title(title: str) -> str:
    """"Microprocessors & Interfacing" == "Microprocessors and interfacing"."""
    return " ".join(re.findall(r"[a-z0-9]+", title.lower().replace("&", " and ")))


def section_key(course: Course) -> tuple:
    return tuple(sorted((s.type, s.section_id, str(sorted(s.timings.items())), s.room, tuple(s.instructors))
                        for offering in course.offerings.all() for s in offering.sections.all() if not s.cancelled))


def known_pairs(codes: set[str]) -> set[frozenset]:
    """Listed equivalents and timetable cross-listings, both courses comparable."""
    known = {frozenset((row.course.code, row.equivalent_code)) for row in
             CourseEquivalent.objects.exclude(source="content").select_related("course")}
    by_class = defaultdict(list)
    for course in Course.objects.filter(code__in=codes).prefetch_related("offerings__sections"):
        key = section_key(course)
        if key:
            by_class[(course.title.strip().lower(), key)].append(course.code)
    known |= {frozenset(pair) for group in by_class.values() for pair in combinations(group, 2)}
    return {pair for pair in known if len(pair) == 2 and pair <= codes}


def scores(found: dict[frozenset, float], known: set[frozenset], unlabelled: set[frozenset],
           threshold: float) -> tuple[float, float, float]:
    predicted = {pair for pair, value in found.items() if value >= threshold} - unlabelled
    hits = len(predicted & known)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(known) if known else 0.0
    return precision, recall, (2 * precision * recall / (precision + recall) if hits else 0.0)


class Command(BaseCommand):
    help = "Choose the same-class detection thresholds from the listed equivalents."

    def handle(self, *args, **options) -> None:
        index = get_piece_index()
        codes = comparable_codes(index)
        known = known_pairs(codes)
        titles = dict(Course.objects.values_list("code", "title"))
        same_title = lambda pair: len({normal_title(titles.get(code, "")) for code in pair}) == 1  # noqa: E731
        grid = []
        for measure in MEASURES:
            for twin in TWIN_SIMS:
                found = overlaps(index, codes, twin, measure)
                for pair in separate_classes(set(found)):  # the app drops these too (equivalents.detect)
                    found[pair] = 0.0
                unlabelled = {pair for pair in found if pair not in known and same_title(pair)}
                for threshold in THRESHOLDS:
                    grid.append((*scores(found, known, unlabelled, threshold), twin, threshold, found, measure))
        best = max(grid, key=lambda row: (round(row[2], 4), row[4], row[3]))
        precision, recall, f1, twin, threshold, found, measure = best
        name = lambda pair: " = ".join(f"{code} {titles.get(code, '')[:34]}" for code in sorted(pair))  # noqa: E731
        missed = sorted(((found.get(pair, 0.0), pair) for pair in known if found.get(pair, 0.0) < threshold), key=lambda row: row[0])
        unlabelled = sorted(((value, pair) for pair, value in found.items() if value >= threshold and pair not in known
                             and same_title(pair)), key=lambda row: -row[0])
        wrong = sorted(((value, pair) for pair, value in found.items() if value >= threshold and pair not in known
                        and not same_title(pair)), key=lambda row: -row[0])
        lines = [
            "# Same-class detection EDA", "",
            f"Generated by `python manage.py equivalence_eda`. {len(codes)} comparable courses (project courses and title-only "
            f"courses left out), {len(known)} known same-class pairs (listed equivalents + timetable cross-listings).", "",
            f"**Chosen: EQUIVALENT_MEASURE = \"{measure}\", EQUIVALENT_TWIN_SIM = {twin}, EQUIVALENT_OVERLAP = {threshold}** "
            f"(precision {precision:.2f}, recall {recall:.2f}, F1 {f1:.2f}; config.py has {config.EQUIVALENT_MEASURE} / "
            f"{config.EQUIVALENT_TWIN_SIM} / {config.EQUIVALENT_OVERLAP}).", "",
            "Rule: best F1 on labelled pairs, ties to the higher threshold. Recall can't reach 1: some listed equivalents "
            "are regulations, not the same text (CE F420 / CE G616 Bridge Engineering share no sentence).", "",
            "| measure | twin sim | overlap >= | precision | recall | F1 |", "|---|---|---|---|---|---|",
            *[f"| {m} | {t} | {h} | {p:.2f} | {r:.2f} | {f:.2f} |" for p, r, f, t, h, _, m in grid if h in (0.5, 0.7, 0.8, 0.9, 0.95)],
            "", f"## Known pairs below the threshold ({len(missed)})", "",
            *[f"- {value:.2f}  {name(pair)}" for value, pair in missed[:SHOWN]],
            "", f"## Different titles at or above the threshold ({len(wrong)}): counted as wrong", "",
            *[f"- {value:.2f}  {name(pair)}" for value, pair in wrong[:SHOWN]],
            "", f"## Same title, not listed, at or above the threshold ({len(unlabelled)}): unscored, check by hand", "",
            *[f"- {value:.2f}  {name(pair)}" for value, pair in unlabelled[:SHOWN]],
        ]
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n")
        self.stdout.write("\n".join(lines[4:6]) + f"\nReport: {OUT}")
