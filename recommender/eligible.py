"""The get_eligible_courses pipeline, one small function per stage.

A: rule filters (offered, category, not done, prerequisites, batch, exclude)  ->  B: handout filters
->  C: retrieve (embeddings) + rerank (cross-encoder, recommender/ranking.py) + relevance cutoff + profile boost +
dislike penalties, then sort  ->  D: timetable preferences  ->  fit with the current courses  ->  E: result.
Every removal is recorded in `Pipeline.removed` (stage -> codes), for tests and the debug panel.
"""
import re
from dataclasses import dataclass, field
from functools import cmp_to_key

import numpy as np
from django.db.models import Prefetch

from catalog.models import Course, Offering, Rule
from students.models import EVAL_STYLES
from students.templatetags.text import course_title

from . import config
from .codes import normalise_code, resolve_course
from .context import StudentContext, resolve_preferences
from .embeddings import get_embedder, get_reranker
from .handout_facts import CourseFacts, all_course_facts, get_course_facts, number
from .history import same_class_groups, with_equivalents
from .piece_index import get_piece_index
from .ranking import Scored, closest, neighbours, rerank, retrieve
from .plan import check_plan
from .timetable import timetable_term

# filter key -> (CourseFacts field, pass test given the fact value and the filter's argument)
FILTERS = {
    "no_midsem": ("has_midsem", lambda value, _: value is False),
    "no_attendance": ("attendance_required", lambda value, _: value is False),
    "lenient_makeup": ("makeup_lenient", lambda value, _: value is True),
    "project_based": ("project_percent", lambda value, _: value > 0),
    "open_book": ("open_book", lambda value, _: value is True),
    "max_quizzes": ("quiz_count", lambda value, limit: value <= limit),
    "max_compre_percent": ("compre_percent", lambda value, limit: value <= limit),
    "min_project_percent": ("project_percent", lambda value, limit: value >= limit),
}
COUNTS_AS_NOTE = "counts as an OPEL, since your {category} requirement is complete"
HIGHER_DEGREE_NOTE = "higher-degree course: max 1 per semester, needs a minimum CGPA (set by the AGC, not published)"
PREREQ_NOTE = "prerequisites couldn't be verified"
TIMINGS_NOTE = "some class times aren't listed in the timetable"
SOP_NOTE = ("You're planning an SOP: keep a slot free for it and decide with a professor.")
SECTION_TYPES = ("lecture", "tutorial", "practical")
SEARCH_ORDER = ("HUEL", "DEL", "OPEL")


@dataclass
class Candidate:
    """A course still in the running, and everything the result says about it."""

    course: Course
    category: str
    facts: CourseFacts
    sections: list = field(default_factory=list)   # non-cancelled sections this semester
    counts_as: str | None = None
    notes: list[str] = field(default_factory=list)
    filters_passed: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    unverified_notes: list[str] = field(default_factory=list)
    embedding: float | None = None                  # best piece similarity (C1)
    relevance: float | None = None                  # cross-encoder relevance, 0-1 (C2); a neighbour's content similarity
    penalty: float = 0.0
    personal: float = 0.0                           # profile boost (apply_profile); negative for "struggled with"
    personal_why: list[str] = field(default_factory=list)
    matched_on: str | None = None
    matched_topic: str | None = None                # the student's topic that matched_on answered
    by_topic: list[float] = field(default_factory=list)  # relevance per topic
    neighbour_of: int | None = None                 # no direct match: added for this topic by content (add_neighbours)
    similar_to: str = ""                            # the anchors a neighbour is close to ("GS F343 Short Film…")
    why: str = ""
    rank: int = 0                                   # 1-based, before stage D
    picked_sections: dict = field(default_factory=dict)  # {"lecture": "L2"}: a fit with the current courses
    same_class: list["Candidate"] = field(default_factory=list)  # other codes of this very class (ECE F434 for EEE F434)

    @property
    def code(self) -> str:
        return self.course.code

    @property
    def final(self) -> float:
        return (self.relevance or 0) + self.penalty + self.personal


@dataclass
class Pipeline:
    """Resolved settings, the working lists and what each stage did."""

    ctx: StudentContext
    settings: dict
    searched: list[str]
    exclude: set[str]
    filters: dict
    warnings: list[str] = field(default_factory=list)
    removed: dict[str, list[str]] = field(default_factory=dict)
    excluded: list[dict] = field(default_factory=list)
    offered_codes: set[str] = field(default_factory=set)
    catalogue: list[Scored] = field(default_factory=list)       # the whole catalogue's best matches, reranked
    next_semester_relevance: dict[str, float] = field(default_factory=dict)
    next_semester: list[tuple[str, str]] = field(default_factory=list)  # (code, "X needs Y, which you're taking…")
    topics: list[str] = field(default_factory=list)  # the student's topics (or profile interests)
    branches: list[str] = field(default_factory=list)  # Course.department codes named as a subject ("maths")

    def remove(self, stage: str, code: str) -> None:
        self.removed.setdefault(stage, []).append(code)


# ---------------------------------------------------------------- settings

def resolve_exclude(codes: list[str] | None) -> tuple[set[str], list[str]]:
    """Normalised codes plus their equivalents, and a warning per unknown code."""
    found, warnings = set(), []
    for raw in codes or []:
        code = normalise_code(raw)
        course = resolve_course(code)
        if course is None:
            warnings.append(f"Unknown course in exclude: {code}")
            continue
        found |= {code, course.code}
    return with_equivalents(found), warnings


def max_extra_electives() -> int | None:
    rule = Rule.objects.filter(rule_id="reg_extra_electives").first()
    return rule.values.get("max_extra_electives") if rule else None


def plural(categories: list[str]) -> str:
    """["HUEL", "DEL"] -> "HUELs or DELs"."""
    return " or ".join(f"{category}s" for category in categories)


def split_branches(about: list[str] | None) -> tuple[list[str], list[str]]:
    """(department codes, remaining topics). A topic that is only a branch word (config.BRANCH_ALIASES, filler like
    "courses" ignored) becomes a department filter; anything longer ("financial markets", "maths for ML") stays a topic.
    why whole-topic only: matching words inside topics would turn "financial markets" into a finance filter."""
    branches, topics = [], []
    for topic in about or []:
        key = " ".join(word for word in re.findall(r"[a-z]+", topic.lower()) if word not in config.BRANCH_FILLER)
        codes = config.BRANCH_ALIASES.get(key)
        if codes:
            branches += [code for code in codes if code not in branches]
        else:
            topics.append(topic)
    return branches, topics


# ---------------------------------------------------------------- which categories

def categories_to_search(ctx: StudentContext, category: str | None) -> tuple[list[str], list[str], bool]:
    """(categories, warnings, stop). stop = everything is complete, so there's nothing to search."""
    if category:
        return [category], [], False
    if ctx.known_gap:
        return list(SEARCH_ORDER), [ctx.known_gap], False
    open_ones = [name for name in SEARCH_ORDER if not ctx.remaining[name].complete]
    if open_ones:
        return open_ones, [], False
    extra = max_extra_electives()
    return [], ["Your HUEL, DEL and OPEL requirements are done. More electives would be extras"
                + (f" (at most {extra} allowed)." if extra is not None else ".")], True


def complete_warnings(ctx: StudentContext, category: str) -> list[str]:
    """Explicit search for a requirement that's already done: say what the extra courses will count as."""
    status = ctx.remaining.get(category)
    if not status or not status.complete:
        return []
    extra = max_extra_electives()
    extra_text = f"will be extra electives (at most {extra} extra allowed)." if extra is not None else "will be extra electives."
    if status.required_courses == 0:
        return [f"Your programme has no {category} requirement, so these {extra_text}"]
    opel_done = ctx.remaining["OPEL"].complete
    tail = f"Extra {category}s {extra_text}" if category == "OPEL" or opel_done else f"Extra {category}s will count as OPELs instead."
    return [f"Your {category} requirement is already complete ({status.done_units} of {status.required_units} units). {tail}"]


# ---------------------------------------------------------------- stage A

def offered_courses(semester_tag: str) -> list[Course]:
    """Courses with at least one non-cancelled section this semester, sections prefetched."""
    live = Offering.objects.filter(semester_tag=semester_tag).prefetch_related("sections").order_by("pk")
    courses = (Course.objects.filter(offerings__semester_tag=semester_tag, offerings__sections__cancelled=False)
               .distinct().order_by("code").prefetch_related(Prefetch("offerings", queryset=live)))
    return list(courses)


def live_sections(course: Course) -> list:
    """Non-cancelled sections of the course's first offering with any (the prefetched semester offerings)."""
    for offering in course.offerings.all():
        sections = [section for section in offering.sections.all() if not section.cancelled]
        if sections:
            return sections
    return []


def category_of(pipeline: Pipeline, code: str) -> tuple[str, str | None] | None:
    """(category, counts_as) if the course belongs to a searched category; None otherwise.
    OPEL also takes DELs / HUELs once that requirement is complete."""
    ctx, category = pipeline.ctx, pipeline.ctx.category_map.get(code)
    if category in pipeline.searched:
        return category, None
    if "OPEL" in pipeline.searched and category in ("DEL", "HUEL") and category in ctx.remaining \
            and ctx.remaining[category].complete:
        return category, "OPEL"
    return None


def prerequisite_state(course: Course, completed: set[str], current: set[str]) -> str:
    """"met", "unknown" (flagged / unparseable), "next_semester" (met once current courses are done) or "unmet".
    Only completed courses count; a current course satisfies a prerequisite next semester."""
    if course.needs_verification and not course.prerequisites:
        return "unknown"
    groups = course.prerequisites or []
    if all(set(group) & completed for group in groups):
        return "met"
    if all(set(group) & (completed | current) for group in groups):
        return "next_semester"
    return "unmet"


def newer_batch_only(ctx: StudentContext, course: Course, term: tuple[int, int] | None) -> bool:
    """A course only the batch admitted in the timetable's year may take (com code rule), for an older student."""
    return course.only_2026_batch and term is not None and ctx.student.admission_year < term[0]


def stage_a(pipeline: Pipeline) -> list[Candidate]:
    """Rule filters, in order: category, branch, not done, prerequisites, batch, exclude. Records removals and warnings."""
    ctx = pipeline.ctx
    counts: dict[str, int] = {}
    candidates = []
    all_course_facts()  # why: fills the facts cache in two queries instead of one per course
    term = timetable_term()  # why once: it's a query, and this loop runs over every offered course
    offered = offered_courses(ctx.semester_tag)
    pipeline.offered_codes = {course.code for course in offered}
    for course in offered:
        placed = category_of(pipeline, course.code)
        if placed is None:
            continue
        counts["category"] = counts.get("category", 0) + 1
        if pipeline.branches and course.department not in pipeline.branches:
            pipeline.remove("branch", course.code)
            continue
        counts["branch"] = counts.get("branch", 0) + 1
        if course.code in ctx.completed or course.code in ctx.current:
            pipeline.remove("done", course.code)
            continue
        counts["done"] = counts.get("done", 0) + 1
        state = prerequisite_state(course, ctx.completed, ctx.current)
        if state in ("unmet", "next_semester"):
            pipeline.remove("prerequisites", course.code)
            if state == "next_semester":
                needed = sorted({code for group in course.prerequisites for code in group if code in ctx.current
                                 and not set(group) & ctx.completed})
                pipeline.next_semester.append((course.code, f"{course.code} needs {' and '.join(needed)} (you're taking it now), so you can take it next semester"))
            continue
        counts["prerequisites"] = counts.get("prerequisites", 0) + 1
        if newer_batch_only(ctx, course, term):
            pipeline.remove("batch", course.code)
            continue
        counts["batch"] = counts.get("batch", 0) + 1
        if course.code in pipeline.exclude:
            pipeline.remove("exclude", course.code)
            continue
        candidate = Candidate(course=course, category=placed[0], counts_as=placed[1], facts=get_course_facts(course),
                              sections=live_sections(course))
        if placed[1]:
            candidate.notes.append(COUNTS_AS_NOTE.format(category=placed[0]))
        if state == "unknown":
            candidate.notes.append(PREREQ_NOTE)
        if course.is_higher_degree:
            candidate.notes.append(HIGHER_DEGREE_NOTE)
        candidates.append(candidate)
    if not candidates:
        pipeline.warnings.append(empty_reason(pipeline, counts))
    return candidates


def empty_reason(pipeline: Pipeline, counts: dict[str, int]) -> str:
    """Which stage removed everything, in one sentence."""
    kinds = plural(pipeline.searched)
    if not counts.get("category"):
        return f"No {kinds} are offered this semester."
    if not counts.get("branch"):
        return f"No {kinds} from {', '.join(pipeline.branches)} are offered this semester."
    if not counts.get("done"):
        return f"No {kinds} you haven't already taken are offered this semester."
    if not counts.get("prerequisites"):
        return f"Every {kinds[:-1] if len(pipeline.searched) == 1 else 'course'} you haven't taken yet needs prerequisites you haven't completed."
    if not counts.get("batch"):
        return f"The remaining {kinds} are only open to the newest batch."
    return f"Every remaining {kinds[:-1] if len(pipeline.searched) == 1 else 'course'} is in your exclude list."


# ---------------------------------------------------------------- stage B

def stage_b(pipeline: Pipeline, candidates: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    """(passed, couldnt_verify). A failed filter removes the course; an unknown one sends it to couldnt_verify."""
    active = {key: value for key, value in pipeline.filters.items() if value is not False}
    passed, unknown = [], []
    for candidate in candidates:
        failed = False
        for key, value in active.items():
            fact_field, test = FILTERS[key]
            fact = getattr(candidate.facts, fact_field)
            if fact is None:
                candidate.unverified.append(key)
                note = candidate.facts.note_for(fact_field) or "not stated in the handout"
                if note not in candidate.unverified_notes:
                    candidate.unverified_notes.append(note)
            elif test(fact, value):
                candidate.filters_passed.append(key)
            else:
                failed = True
        if failed:
            pipeline.remove("filters", candidate.code)
        elif candidate.unverified:
            unknown.append(candidate)
        else:
            passed.append(candidate)
    return passed, unknown


# ---------------------------------------------------------------- stage C

def query_topics(ctx: StudentContext, about: list[str] | None) -> tuple[list[str], str]:
    """(topics to embed, ranked_by). Each topic is scored on its own."""
    if about:
        return list(about), "about"
    if ctx.interests:
        return list(ctx.interests), "profile_interests"
    return [], "handout_completeness"


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def penalties(facts: CourseFacts, styles: list[str]) -> list[str]:
    """One explanation per disliked evaluation style the course has (unknown facts never count)."""
    found = {
        "many_quizzes": facts.quiz_count is not None and facts.quiz_count >= config.MANY_QUIZZES_THRESHOLD
        and f"{facts.quiz_count} quizzes (you'd rather avoid many)",
        "closed_book": facts.open_book is False and "closed-book exams (you'd rather avoid them)",
        "strict_attendance": facts.attendance_required is True and "attendance required (you'd rather avoid it)",
        "heavy_compre": facts.compre_percent is not None and facts.compre_percent >= config.HEAVY_COMPRE_PERCENT
        and f"compre is {number(facts.compre_percent)}% (you'd rather avoid a heavy compre)",
        "no_makeup": facts.makeup_allowed is False and "no makeups (you'd rather avoid that)",
    }
    return [found[style] for style, _ in EVAL_STYLES if style in styles and found[style]]


def matched_text(index, row: int) -> str:
    """"topic: Transformers and attention", cut to MATCHED_ON_MAX_CHARS."""
    return shorten(f"{index.kinds[row].replace('_', ' ')}: {index.texts[row]}", config.MATCHED_ON_MAX_CHARS)


def relevance_scores(pipeline: Pipeline, ranked_by: str, passed: list[Candidate],
                     unknown: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    """C1 + C2. Each list keeps only its top RERANK_CANDIDATES by embedding, now with embedding / relevance /
    matched_on. The whole catalogue (topic searches: anchors for neighbours, better matches not offered) and the
    "next semester" courses are reranked in the same call."""
    index, topics = get_piece_index(), pipeline.topics
    query = get_embedder().embed(topics, kind="query")
    main = retrieve(index, [candidate.code for candidate in passed], query, config.RERANK_CANDIDATES)
    unverified = retrieve(index, [candidate.code for candidate in unknown], query, config.RERANK_CANDIDATES)
    catalogue = retrieve(index, list(index.rows_by_course), query, config.RERANK_CANDIDATES) if ranked_by == "about" else []
    later = retrieve(index, [code for code, _ in pipeline.next_semester], query, len(pipeline.next_semester))
    rerank([main, unverified, catalogue, later], topics, query, index, get_reranker())
    pipeline.catalogue = catalogue
    pipeline.next_semester_relevance = {item.code: item.relevance for item in later}

    def attach(candidates: list[Candidate], scored: list[Scored], stage: str) -> list[Candidate]:
        by_code = {item.code: item for item in scored}
        kept = []
        for candidate in candidates:
            item = by_code.get(candidate.code)
            if item is None:
                pipeline.remove(stage, candidate.code)
                continue
            topic = int(np.argmax(item.by_topic))
            candidate.embedding, candidate.by_topic, candidate.relevance = item.embedding, item.by_topic, item.relevance
            candidate.matched_on = matched_text(index, item.best_rows[topic])
            candidate.matched_topic = topics[topic]
            kept.append(candidate)
        return kept
    return attach(passed, main, "retrieval"), attach(unknown, unverified, "retrieval")


def profile_phrases(ctx: StudentContext, ranked_by: str) -> list[tuple[str, str]]:
    """(phrase to search, why text) for each interest and each strengths phrase.
    why split the strengths on commas / "and": probe 2026-09-28, "programming and algorithms" as one phrase found only
    Data Structures and Algorithms; "programming" + "algorithms" found OOP, Computer Programming and PL too."""
    # why no interests when they ARE the query: they'd count twice
    phrases = [(interest, f"matches your interest in {interest}") for interest in
               ([] if ranked_by == "profile_interests" else ctx.interests)]
    for part in re.split(r"[,;/&\n]|\band\b", ctx.strengths):
        if part.strip():
            phrases.append((part.strip(), f"matches your strength '{part.strip()}'"))
    return phrases


def text_boosts(ctx: StudentContext, ranked_by: str, codes: list[str]) -> dict[str, tuple[float, str]]:
    """code -> (boost, why) from the profile phrases, scored exactly like a topic (retrieve + rerank); only real
    matches (relevance >= RELEVANCE_CUTOFF) count, by their best phrase."""
    found, searched = {}, []
    for phrase, why in profile_phrases(ctx, ranked_by):
        # a branch word ("maths") is a department, as in a query; the search can't match the abbreviation
        departments, _ = split_branches([phrase])
        for code in codes if departments else []:
            if code.split()[0] in departments and code not in found:
                found[code] = (config.PROFILE_TEXT_WEIGHT, why)
        if not departments:
            searched.append((phrase, why))
    if not searched or not codes:
        return found
    index, topics = get_piece_index(), [phrase for phrase, _ in searched]
    vectors = get_embedder().embed(topics, kind="query")
    scored = retrieve(index, codes, vectors, config.RERANK_CANDIDATES)
    rerank([scored], topics, vectors, index, get_reranker())
    for item in scored:
        best = int(np.argmax(item.by_topic))
        boost = config.PROFILE_TEXT_WEIGHT * item.by_topic[best]
        if item.by_topic[best] >= config.RELEVANCE_CUTOFF and boost > found.get(item.code, (0.0, ""))[0]:
            found[item.code] = (boost, searched[best][1])
    return found


def course_parts(ctx: StudentContext) -> list[tuple[dict, float, str]]:
    """(label -> course vector, weight, why text) for the did well / struggled with courses."""
    index = get_piece_index()
    titles = dict(Course.objects.filter(code__in=ctx.did_well + ctx.struggled).values_list("code", "title"))

    def courses(codes: list[str]) -> dict:
        return {f"{code} {course_title(titles.get(code, ''))}": index.course_vectors[code]
                for code in codes if code in index.course_vectors}
    well_weight = config.PROFILE_WEIGHT * (config.GRADE_ORIENTED_FACTOR if ctx.grade_oriented else 1)
    return [(vectors, weight, text) for vectors, weight, text in
            ((courses(ctx.did_well), well_weight, "close to {}, which you did well in"),
             (courses(ctx.struggled), -config.PROFILE_WEIGHT, "close to {}, which you struggled with")) if vectors]


def apply_profile(ctx: StudentContext, ranked_by: str, candidates: list[Candidate]) -> None:
    """C4a: interests / strengths add PROFILE_TEXT_WEIGHT x relevance when a phrase really matches the course (the
    topic search); did well / struggled with add weight x (cosine - floor) for the closest such course, when above the
    floor (stricter with a topic, see config). It never removes a course, and the cutoff (C3) already ran on relevance
    alone, so with a topic it only reorders real matches."""
    texts = text_boosts(ctx, ranked_by, [candidate.code for candidate in candidates])
    parts = course_parts(ctx)
    floor = config.PROFILE_FLOOR_TOPIC if ranked_by == "about" else config.PROFILE_FLOOR
    for candidate in candidates:
        if candidate.code in texts:
            boost, why = texts[candidate.code]
            candidate.personal += boost
            candidate.personal_why.append(why)
        for vectors, weight, text in parts:
            found = closest(get_piece_index(), candidate.code, vectors)
            if found and found[1] >= floor:
                candidate.personal += weight * (found[1] - floor)
                candidate.personal_why.append(text.format(found[0]))


def apply_penalties(ctx: StudentContext, ranked_by: str, has_query: bool, candidates: list[Candidate]) -> None:
    """C4: dislike penalties and the `why` text (built here, never by the LLM)."""
    for candidate in candidates:
        reasons = penalties(candidate.facts, ctx.avoid_eval_styles)
        candidate.penalty = -config.DISLIKE_PENALTY * len(reasons)
        parts = []
        if candidate.similar_to:
            parts.append(f"no direct match for '{candidate.matched_topic}'; its content is close to {candidate.similar_to}")
        elif candidate.matched_on and ranked_by == "about":
            parts.append(f"matches {candidate.matched_on}")
        if ranked_by != "about":
            candidate.matched_topic = None  # why: profile interests aren't "your topic", same reason as below
        if ranked_by == "profile_interests":
            # why no matched piece here: the model turned "matches topic: X" into "your interest in X"
            related = candidate.relevance is not None and candidate.relevance >= config.RELEVANCE_CUTOFF
            parts.append("fits your interests" if related else "not close to your interests")
        parts += candidate.personal_why + reasons
        if has_query and not candidate.facts.has_handout:
            parts.append("no handout; matched on the Bulletin text only")
        candidate.why = "; ".join(parts)


def anchors(pipeline: Pipeline, topic: int) -> list[str]:
    """The catalogue's best matches for the topic (offered or not), if any is at least NEIGHBOUR_ANCHOR_FLOOR."""
    best = sorted(pipeline.catalogue, key=lambda item: (-item.by_topic[topic], item.code))[:config.NEIGHBOUR_ANCHORS]
    return [item.code for item in best if item.by_topic[topic] >= config.NEIGHBOUR_ANCHOR_FLOOR]


def add_neighbours(pipeline: Pipeline, eligible: list[Candidate], real: list[Candidate], places: int) -> list[Candidate]:
    """For each topic with fewer real matches than `places`: the eligible courses closest in content to its anchors,
    up to the missing number (at most NEIGHBOURS_PER_TOPIC). They rank after every direct match (compare) and say what they're close to."""
    taken, added = {candidate.code for candidate in real}, []
    for topic, name in enumerate(pipeline.topics):
        direct = sum(on_topic(candidate, topic) for candidate in real)
        found = anchors(pipeline, topic) if direct < places else []
        titles = dict(Course.objects.filter(code__in=found).values_list("code", "title"))
        pool = {candidate.code: candidate for candidate in eligible if candidate.code not in taken}
        close = [(code, similarity) for code, similarity in (neighbours(get_piece_index(), found, list(pool)) if found else [])
                 if similarity >= config.NEIGHBOUR_MIN_SIMILARITY][:min(places - direct, config.NEIGHBOURS_PER_TOPIC)]
        for code, similarity in close:
            candidate = pool[code]
            candidate.relevance, candidate.neighbour_of, candidate.matched_topic = similarity, topic, name
            candidate.similar_to = ", ".join(f"{anchor} {course_title(titles.get(anchor, ''))}" for anchor in found)
            added.append(candidate)
            taken.add(code)
        if not direct:
            pipeline.warnings.append(f"Nothing this semester matches '{name}' directly"
                                     + ("; the ones listed are the closest in content."
                                        if close else "."))
    return added


def relevance_cutoff(pipeline: Pipeline, eligible: list[Candidate], passed: list[Candidate], unknown: list[Candidate],
                     places: int) -> tuple[list[Candidate], list[Candidate]]:
    """C3 (topic searches only): keep real matches (relevance before penalties >= RELEVANCE_CUTOFF), then top up
    each short topic with neighbours from all `eligible` courses (retrieved or not)."""
    real = [candidate for candidate in passed if candidate.relevance >= config.RELEVANCE_CUTOFF]
    for candidate in passed + unknown:
        if candidate.relevance < config.RELEVANCE_CUTOFF:
            pipeline.remove("cutoff", candidate.code)
    if eligible:  # why: with nothing eligible, stage A already said why
        real += add_neighbours(pipeline, eligible, real, places)
    return real, [candidate for candidate in unknown if candidate.relevance >= config.RELEVANCE_CUTOFF]


def stage_c(pipeline: Pipeline, ranked_by: str, passed: list[Candidate], unknown: list[Candidate],
            places: int) -> tuple[str, list[Candidate], list[Candidate]]:
    """Retrieve, rerank, cut off (+ neighbours), penalise. Returns (ranked_by, passed, unknown)."""
    if pipeline.topics:
        scored_passed, scored_unknown = relevance_scores(pipeline, ranked_by, passed, unknown)
        if ranked_by == "about":
            scored_passed, scored_unknown = relevance_cutoff(pipeline, passed, scored_passed, scored_unknown, places)
        if ranked_by == "profile_interests" and not any(candidate.relevance >= config.RELEVANCE_CUTOFF
                                                        for candidate in scored_passed + scored_unknown):
            # nothing relates to the interests (e.g. HUELs for an ML student): a relevance score would be noise
            for candidate in passed + unknown:
                candidate.embedding = candidate.relevance = candidate.matched_on = candidate.matched_topic = None
            pipeline.removed.pop("retrieval", None)
            pipeline.topics, ranked_by = [], "handout_completeness"
        else:
            passed, unknown = scored_passed, scored_unknown
    apply_profile(pipeline.ctx, ranked_by, passed + unknown)
    apply_penalties(pipeline.ctx, ranked_by, bool(pipeline.topics), passed + unknown)
    return ranked_by, passed, unknown


def next_semester_warnings(pipeline: Pipeline, has_query: bool) -> list[str]:
    """"Eligible next semester" notes: with a query, only courses that pass the cutoff (most relevant first)."""
    notes = pipeline.next_semester
    if has_query:
        relevance_of = pipeline.next_semester_relevance
        notes = sorted(((code, note) for code, note in notes if relevance_of.get(code, 0) >= config.RELEVANCE_CUTOFF),
                       key=lambda item: (-relevance_of[item[0]], item[0]))
    return [note for _, note in notes[:config.MAX_NEXT_SEMESTER_WARNINGS]]


def unoffered_matches(pipeline: Pipeline, shown: list[Candidate]) -> list[dict]:
    """Real matches not offered this semester that beat everything shown (so the student hears "System Security
    isn't offered" instead of nothing): catalogue courses of the searched categories the student hasn't done."""
    ctx = pipeline.ctx
    best = max((candidate.relevance for candidate in shown if not candidate.similar_to), default=0.0)
    better = [item for item in pipeline.catalogue
              if item.code not in pipeline.offered_codes and item.code not in ctx.completed and item.code not in ctx.current
              and ctx.category_map.get(item.code) in pipeline.searched
              and (not pipeline.branches or item.code.split()[0] in pipeline.branches)
              and item.relevance >= config.RELEVANCE_CUTOFF and item.relevance >= best + config.TIE_EPSILON]
    top = sorted(better, key=lambda item: (-item.relevance, item.code))[:config.MAX_NOT_OFFERED]
    titles = dict(Course.objects.filter(code__in=[item.code for item in top]).values_list("code", "title"))
    return [{"code": item.code, "title": course_title(titles.get(item.code, "")),
             "category": ctx.category_map[item.code]} for item in top]


# ---------------------------------------------------------------- one class, several codes

def merge_same_class(candidates: list[Candidate]) -> list[Candidate]:
    """One candidate per class taught under several codes (EEE F434 = ECE F434): the code that counts best for the
    student (DEL, then OPEL, then HUEL: CATEGORY_TIE_ORDER, with DELs that count as OPELs as OPELs), then the more
    relevant one. The other codes ride along in `same_class`, so the student can pick one of them instead."""
    by_code = {candidate.code: candidate for candidate in candidates}
    kept = []
    for group in same_class_groups([candidate.code for candidate in candidates]):
        members = sorted((by_code[code] for code in group),
                         key=lambda candidate: (tie_position(candidate), -(candidate.relevance or 0), candidate.code))
        best = members[0]
        best.same_class = members[1:]
        if best.same_class:
            others = ", ".join(f"{other.code} ({other.counts_as or other.category})" for other in best.same_class)
            best.notes.append(f"the same class is also offered as {others}; take only one")
        kept.append(best)
    return kept


# ---------------------------------------------------------------- sort

def tie_position(candidate: Candidate) -> int:
    category = candidate.counts_as or candidate.category
    return config.CATEGORY_TIE_ORDER.index(category) if category in config.CATEGORY_TIE_ORDER else len(config.CATEGORY_TIE_ORDER)


def compare(ranked_by: str):
    """Direct matches before neighbours, lecture courses before projects, then final score (ties within TIE_EPSILON),
    then category order, then completeness, then code. With no query every final is just the penalty, so completeness comes before the category order."""
    def key_order(a: Candidate, b: Candidate) -> int:
        if (a.relevance is None) != (b.relevance is None) and ranked_by != "handout_completeness":
            return 1 if a.relevance is None else -1
        if (a.neighbour_of is None) != (b.neighbour_of is None):  # a direct match beats a neighbour
            return 1 if a.neighbour_of is not None else -1
        # why: a study / lab / design project is arranged with a professor, not picked like a lecture course
        if a.course.is_project_course != b.course.is_project_course:
            return 1 if a.course.is_project_course else -1
        if abs(a.final - b.final) >= config.TIE_EPSILON:
            return -1 if a.final > b.final else 1
        keys = [(tie_position(a), tie_position(b)), (-a.facts.completeness, -b.facts.completeness)]
        if ranked_by == "handout_completeness":
            keys.reverse()
        for left, right in keys + [(a.code, b.code)]:
            if left != right:
                return -1 if left < right else 1
        return 0
    return cmp_to_key(key_order)


def rank(candidates: list[Candidate], ranked_by: str) -> list[Candidate]:
    """Sorted, with each candidate's 1-based rank (before stage D)."""
    ordered = sorted(sorted(candidates, key=lambda candidate: candidate.code), key=compare(ranked_by))
    for position, candidate in enumerate(ordered, 1):
        candidate.rank = position
    return ordered


# ---------------------------------------------------------------- stage D

def section_problem(section, avoid_8am: bool, avoid_day: str | None) -> list[str]:
    """Which preferences this section breaks: "8am" and / or "day"."""
    problems = []
    if avoid_8am and any(set(periods) & config.EARLY_PERIODS for periods in section.timings.values()):
        problems.append("8am")
    if avoid_day and section.timings.get(avoid_day):
        problems.append("day")
    return problems


def preference_reason(candidate: Candidate, avoid_8am: bool, avoid_day: str | None) -> str | None:
    """Why the course can't meet the timetable preferences (every section of some type breaks one), or None."""
    reasons = []
    for kind in SECTION_TYPES:
        sections = [section for section in candidate.sections if section.type == kind]
        if not sections or any(not section_problem(section, avoid_8am, avoid_day) for section in sections):
            continue
        broken = {problem for section in sections for problem in section_problem(section, avoid_8am, avoid_day)}
        words = [text for problem, text in (("8am", "has an 8 AM class"), ("day", f"is on {config.DAY_NAMES.get(avoid_day, avoid_day)}"))
                 if problem in broken]
        reasons.append(f"every {kind} section {' or '.join(words)}")
    return "; ".join(reasons) or None


def stage_d(pipeline: Pipeline, ranked: list[Candidate], ranked_by: str) -> list[Candidate]:
    """Timetable preferences are soft ("fewest 8 AMs", like the timetable sort): a course whose every section of
    some type breaks one is kept with a note and ranked a little lower (the same penalty as a disliked style)."""
    avoid_8am, avoid_day = pipeline.settings["avoid_8am"], pipeline.settings["avoid_day"]
    if not avoid_8am and not avoid_day:
        return ranked
    for candidate in ranked:
        if any(not section.timings for section in candidate.sections):
            candidate.notes.append(TIMINGS_NOTE)  # such sections count as OK
        if reason := preference_reason(candidate, avoid_8am, avoid_day):
            candidate.notes.append(reason)
            candidate.penalty -= config.DISLIKE_PENALTY
    return rank(ranked, ranked_by)


def fits(pipeline: Pipeline, candidate: Candidate) -> bool:
    """Does the course fit with the student's current courses (classes, exams, lunch, units, higher degree limit)?
    Keeps the sections check_plan picked; a course that can't fit goes to `excluded` with the reason.
    why here and not only in the LLM's check_plan calls: the model sometimes skips them, and would then recommend
    a course the student can't take."""
    avoid_8am, avoid_day = pipeline.settings["avoid_8am"], pipeline.settings["avoid_day"]
    plan = check_plan(pipeline.ctx, [candidate.code], avoid_8am, avoid_day)
    if plan["ok"] and candidate.code in plan.get("missed_preferences", []) \
            and preference_reason(candidate, avoid_8am, avoid_day) is None:
        # a section avoiding the preference exists, but none of those fits with the current courses: still takeable
        wants = [text for on, text in ((avoid_8am, "an 8 AM class"),
                                       (avoid_day, f"a class on {config.DAY_NAMES.get(avoid_day, avoid_day)}")) if on]
        candidate.notes.append(f"every way it fits with your current courses has {' or '.join(wants)}")
    if plan["ok"]:
        candidate.picked_sections = plan["sections"].get(candidate.code, {})
        return True
    pipeline.remove("fit", candidate.code)
    reason = (f"doesn't fit with your current courses: {plan['problem']}" if plan.get("conflicts")
              else "no combination of its sections fits with your current courses")
    pipeline.excluded.append({"code": candidate.code, "title": course_title(candidate.course.title),
                              "category": candidate.category, "reason": reason})
    return False


def on_topic(candidate: Candidate, topic: int) -> bool:
    """A real match for the student's topic number `topic`, or a neighbour added for it."""
    if candidate.neighbour_of is not None:
        return candidate.neighbour_of == topic
    return len(candidate.by_topic) > topic and candidate.by_topic[topic] >= config.RELEVANCE_CUTOFF


def topic_order(topic: int):
    """Sort key within one topic: direct matches by their score for it, then neighbours by similarity."""
    return lambda candidate: ((1, -candidate.relevance) if candidate.neighbour_of is not None
                              else (0, -candidate.by_topic[topic]))


def stage_fit(pipeline: Pipeline, ranked: list[Candidate], limit: int, topics: int = 0) -> tuple[list[Candidate], int]:
    """Up to `limit` courses that fit with the current courses, in rank order. Returns (kept, how many checked).
    With several topics, each topic first gets up to limit // topics of its own best matches (so "AI and
    electronics" can't come back all AI), then the free places go by rank. Fewer than `topics` places: rank only."""
    checked: dict[str, bool] = {}

    def ok(candidate: Candidate) -> bool:
        if candidate.code not in checked:
            checked[candidate.code] = fits(pipeline, candidate)
        return checked[candidate.code]
    picked: dict[str, Candidate] = {}
    quota = limit // topics if topics > 1 else 0
    for topic in range(topics if quota else 0):
        mine = sum(on_topic(candidate, topic) for candidate in picked.values())
        for candidate in sorted((c for c in ranked if on_topic(c, topic)), key=topic_order(topic)):
            if mine >= quota or len(picked) == limit:
                break
            if candidate.code not in picked and ok(candidate):
                picked[candidate.code] = candidate
                mine += 1
    for candidate in ranked:
        if len(picked) == limit:
            break
        if candidate.code not in picked and ok(candidate):
            picked[candidate.code] = candidate
    return sorted(picked.values(), key=lambda candidate: candidate.rank), len(checked)


def default_count(pipeline: Pipeline) -> int:
    """No number given: MAX_RESULTS per category, and at least one place per topic of the student's."""
    return max(config.MAX_RESULTS, len(pipeline.topics))


def fill(pipeline: Pipeline, ranked: list[Candidate], count: int | None, topics: int) -> tuple[list[Candidate], int]:
    """The courses to return. A number the student gave is the total; without one, each category searched gets
    default_count() places (a DEL that counts as an OPEL fills an OPEL place). Returns (courses, places)."""
    if count or len(pipeline.searched) == 1:
        places = count or default_count(pipeline)
        return stage_fit(pipeline, ranked, places, topics)[0], places
    kept, every_short = [], True
    for category in pipeline.searched:
        pool = [candidate for candidate in ranked if (candidate.counts_as or candidate.category) == category]
        found = stage_fit(pipeline, pool, default_count(pipeline), topics)[0]
        every_short = every_short and len(found) < default_count(pipeline)
        kept += found
    # why: a category that filled its places may have more matches, so "no other course matches" would be false
    places = default_count(pipeline) * len(pipeline.searched) if every_short else len(kept)
    return sorted(kept, key=lambda candidate: candidate.rank), places


# ---------------------------------------------------------------- stage E

def score_object(candidate: Candidate) -> dict:
    # why + 0.0: turns -0.0 into 0.0
    rounded = lambda value: None if value is None else round(value, 2)  # noqa: E731
    return {"embedding": rounded(candidate.embedding), "relevance": rounded(candidate.relevance),
            "penalty": round(candidate.penalty, 2) + 0.0, "final": round(candidate.final, 2), "why": candidate.why,
            **({"personal": round(candidate.personal, 2)} if round(candidate.personal, 2) else {}),
            **({"topic": candidate.matched_topic} if candidate.matched_topic else {}),
            **({"similar_to": candidate.similar_to} if candidate.similar_to else {})}


def shortfall(about: list[str], found: int, count: int | None, wanted: int) -> str | None:
    """Fewer courses than wanted for a topic: say so, so a short list doesn't read as a truncated one."""
    if not 0 < found < wanted:
        return None
    topic = ", ".join(about)
    if count:
        return f"Only {found} of the {count} courses you asked for match '{topic}' and fit your timetable."
    return f"No other course this semester matches '{topic}' and fits your timetable."


def course_entry(ctx: StudentContext, candidate: Candidate, verify: bool = False) -> dict:
    """One course in `courses` / `couldnt_verify`: exact keys, empty optional ones left out."""
    entry = {"code": candidate.code, "title": course_title(candidate.course.title), "category": candidate.category}
    if candidate.counts_as:
        entry["counts_as"] = candidate.counts_as
    if candidate.code in ctx.minor_courses:
        entry["minor"] = ctx.minor_courses[candidate.code]
    notes = candidate.unverified_notes + candidate.notes if verify else candidate.notes
    if verify:
        entry["unverified"] = candidate.unverified
    if notes:
        entry["note"] = "; ".join(notes)
    if candidate.filters_passed:
        entry["filters_passed"] = candidate.filters_passed
    if candidate.picked_sections:
        entry["sections"] = candidate.picked_sections
    if candidate.same_class:
        entry["also_offered_as"] = [{"code": other.code, "title": course_title(other.course.title),
                                     "category": other.counts_as or other.category} for other in candidate.same_class]
    entry["score"] = score_object(candidate)  # why: score.why already says what it matched on
    return entry


def run(ctx: StudentContext, category: str | None, about: list[str] | str | None, filters: dict | None,
        avoid_8am: bool | None, avoid_day: str | None, exclude: list[str] | None,
        count: int | None = None) -> tuple[dict, Pipeline]:
    """The whole pipeline; returns the tool result and the pipeline (for tests / debug)."""
    settings = resolve_preferences(ctx, avoid_8am, avoid_day)
    exclude_codes, warnings = resolve_exclude(exclude)
    searched, search_warnings, stop = categories_to_search(ctx, category)
    pipeline = Pipeline(ctx, settings, searched, exclude_codes, dict(filters or {}), warnings + search_warnings)
    courses, unknown, not_offered, unverified_more = [], [], [], 0
    about = [about] if isinstance(about, str) else about  # one topic given as plain text
    pipeline.branches, about = split_branches(about)
    pipeline.topics, ranked_by = query_topics(ctx, about)
    places = 0
    if not stop:
        if category:
            pipeline.warnings += complete_warnings(ctx, category)
        passed, unknown = stage_b(pipeline, stage_a(pipeline))
        ranked_by, passed, unknown = stage_c(pipeline, ranked_by, passed, unknown, count or default_count(pipeline))
        passed, unknown = merge_same_class(passed), merge_same_class(unknown)
        quota_topics = len(pipeline.topics) if ranked_by == "about" else 0
        courses, places = fill(pipeline, stage_d(pipeline, rank(passed, ranked_by), ranked_by), count, quota_topics)
        ranked_unknown = stage_d(pipeline, rank(unknown, ranked_by), ranked_by)
        unknown, checked = stage_fit(pipeline, ranked_unknown, config.MAX_UNVERIFIED)
        unverified_more = len(ranked_unknown) - checked
        pipeline.warnings += next_semester_warnings(pipeline, ranked_by != "handout_completeness")
        if ranked_by == "about":
            not_offered = unoffered_matches(pipeline, courses)
        if ctx.sop_plan and courses:
            pipeline.warnings.append(SOP_NOTE)
    result = {
        "searched": searched,
        "settings_used": {**settings, "ranked_by": ranked_by, **({"branches": pipeline.branches} if pipeline.branches else {})},
        "courses": [course_entry(ctx, candidate) for candidate in courses],
        "couldnt_verify": [course_entry(ctx, candidate, verify=True) for candidate in unknown],
        "excluded": pipeline.excluded,
        "warnings": pipeline.warnings,
    }
    if unverified_more:
        result["couldnt_verify_more"] = unverified_more  # not checked for fit, only counted
    if not_offered:
        result["better_matches_not_offered"] = not_offered
    if ranked_by == "about" and (short := shortfall(about, len(courses), count, places)):
        result["shortfall"] = short
    return result, pipeline
