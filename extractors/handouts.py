"""Course handouts: dataset/raw/handouts/*.pdf -> dataset/code processed/handouts.json.

Small by design: handles the common handout layouts and flags the rest (needs_verification) instead of
special-casing every odd file. Rules for structured fields; two local models (handout_models.py) for the
attendance / make-up policy wording.

Usage:
    .venv/bin/python extractors/handouts.py                    # all handouts (old output -> handouts_old.json)
    .venv/bin/python extractors/handouts.py 001_AN_F314.pdf …  # only these -> handouts_sample.json
"""
import json
import logging
import multiprocessing
import re
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber

from timetable import PROJECT_ROOT, clean

# ============================== CONFIG ==============================
# One method per field, the same for every handout.
FIELD_METHODS = {
    "course_no": "rule: file name (NNN_DEPT_CODE.pdf)",
    "title / instructors / consultation_hours": "rule: labelled header lines and section headings",
    "about": "rule: description + objectives + outcomes sections; lead-in lines ending in ':' dropped",
    "topics / evaluation": "rule: tables found by their header words",
    "attendance.required, makeup.per_component": "model: embeddings pick the sentences, NLI decides",
    "attendance.percent, makeup.allowed": "rule: a stated '75%'; allowed derived from per_component",
}
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
TOP_K = 3                  # sentences passed from the embedding step to NLI
MIN_SIMILARITY = 0.40      # below this the handout doesn't talk about the policy
NLI_MIN_CONFIDENCE = 0.70  # below this the value is null + an "_uncertain" issue
WEIGHT_TOLERANCE = 1       # evaluation weights must add up to 100 +- this
MIN_TOPICS = 3             # fewer topics from a plan table -> "topics_possibly_truncated"
# (hypothesis, value if entailed, value if contradicted). The negated ones matter: the small model is far
# more decisive on "Make-up is never allowed." than on "Make-ups can be given." for hedged sentences.
HYPOTHESES = {
    "attendance": [("Students are required to attend classes.", True, None), ("A minimum attendance is required.", True, None),
                   ("Attendance is optional.", False, None)],
    "makeup_component": [("A make-up is allowed for the {label}.", True, False), ("There is no make-up for the {label}.", False, True)],
    "makeup_general": [("Make-ups can be given.", True, False), ("Make-up is never allowed.", False, True)],
}
# ====================================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)
for noisy in ("pdfminer", "httpx", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

HANDOUTS_DIR = PROJECT_ROOT / "dataset" / "raw" / "handouts"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "code processed" / "handouts.json"
MANUAL_PATH = PROJECT_ROOT / "dataset" / "manually processed" / "handouts.json"  # typed-in scanned handouts
OVERRIDES_PATH = PROJECT_ROOT / "dataset" / "manually processed" / "handout_overrides.json"  # hand-checked values
TIMETABLE_PATH = PROJECT_ROOT / "dataset" / "code processed" / "timetable.json"

SECTIONS = {  # section -> heading text (the part before ':'), matched at the start of a line
    "description": r"((course|catalog|general) )?description",
    "objectives": r"(course )?(scope|objectives?|aims?)\b.*",
    "outcomes": r"(course )?(learning )?outcomes?",
    "plan": r"(course|lecture) plan.*|modules|plan of work|list of experiments",
    "evaluation": r"evaluation( scheme| schedule| components?)?",
    "attendance": r"attendance( policy)?",
    "makeup": r"make ?-?up( policy)?",
    "consultation": r"(chamber )?consultation.*",
    "other": r"text ?books?.*|reference.*|notices?.*|method of .*|nc .*|criterion.*|grading.*|note.*",
}
# "5. Evaluation:", "I. Course Description:", "1. 2. Scope ...", "2. a) Objective:", "Course Description (Scope and Objective):"
HEADING = re.compile(r"^(?:(?:\d{1,2}(?:\.\d)?|[IVX]{1,4})\s*[.):]?\s*)*(?:[a-z]\)\s*)?([A-Za-z][A-Za-z &/-]{2,40}?)\s*(?:\([^)]*\))?\s*(?::|$)(.*)$")
KINDS = {  # evaluation component name -> kind; first match wins; \b because "tut" is inside "institute"
    "midsem": r"\bmid", "compre": r"\bcompre|\bfinal exam|\bend ?-?sem", "viva": r"\bviva", "tutorial_test": r"\btut",
    "quiz": r"\bquiz|\bclass tests?", "lab": r"\blab|\bpractical|\bexperiment", "assignment": r"\bassign|\bhome ?work",
    "project": r"\bproject", "seminar": r"\bseminar", "presentation": r"\bpresentation", "class_participation": r"\bparticipation|\battendance",
}
LABELS = {"quiz": "quizzes", "midsem": "mid-semester exam", "compre": "comprehensive exam", "assignment": "assignments",
          "project": "project", "lab": "lab components", "tutorial_test": "tutorial tests", "presentation": "presentations",
          "viva": "viva", "seminar": "seminars", "class_participation": "class participation", "other": "other components"}
MAKEUP_WORD = r"make\s*-?\s*ups?"
LIST_MARKER = re.compile(r"^\s*(\(?[a-z]\)|\(?\d{1,2}[.)]|\(?[ivx]{1,4}[.)]|CLO\s*\d+\s*[.:]?|[•▪●○■◦➢✓*\-])\s*", re.I)
# "L2.1-2.2 ", "L3.1-L3.2 ", "L# 22-23: ", "Module 4: ", "3. "
TOPIC_PREFIX = re.compile(r"^\s*((M|L|Lec|Lectures?|Module|Unit|Week)\.?\s*[#-]?\s*\d+(\.\d+)*(\s*(-|–|to)\s*L?\d+(\.\d+)*)?\s*[:.)\-–]?\s*|\d{1,2}\s*[.)]\s*)", re.I)
PRIVATE_BULLET = re.compile(r"[\uf0a7\uf0a8\uf0b7\uf0d8\uf076\uf0fc\uf0e0]\s*")
LEAD_IN = re.compile(r"^.*?\b(?:will|should|shall) be able to\b[:\s,–-]*", re.I)
SENTENCE_END = re.compile(r"(?<=[a-z0-9)\]]{2}[.;?])\s+(?=[A-Za-z(])")
INLINE_LETTER = re.compile(r"\s+(?=[a-hA-H][.)]\s+[A-Z])")  # "... concepts b. To engage ..."

Line = tuple[int, str]


@dataclass
class Document:
    lines: list[Line]                          # (page, text)
    tables: list[tuple[int, list[list[str]]]]  # (page, rows)


@dataclass
class Handout:
    file: str
    course_no: str
    title: str | None
    instructors: list[str]
    about: str | None  # description + objectives + outcomes as one text (kept for search / embeddings)
    topics: list[str]
    evaluation: list[dict]
    attendance: dict
    makeup: dict
    consultation_hours: str | None
    extraction_methods: dict = field(default_factory=dict)
    source_pages: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    needs_verification: bool = False
    # the same content, structured for display
    description: str | None = None
    objectives: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    topic_groups: list[dict] = field(default_factory=list)  # [{"module": str | None, "topics": [str]}]


# ------------------------------ reading
def load_pdf(path: Path) -> Document | None:
    """Text lines and tables of every page; None (logged) if the PDF can't be read."""
    try:
        with pdfplumber.open(path) as pdf:
            pages = [without_overprint(page.dedupe_chars()) for page in pdf.pages]  # dedupe: "fake bold" letters are drawn twice
            lines = [(p.page_number, clean(t)) for p in pages for t in (p.extract_text(x_tolerance=1.5) or "").splitlines() if clean(t)]
            lines = drop_page_furniture(lines)
            tables = [(p.page_number, [[cell_lines(c) for c in row] for row in t]) for p in pages for t in p.extract_tables() if t]
        return Document(lines, tables)
    except Exception:
        log.exception("Could not read %s", path.name)
        return None


def overprinted(chars: list[dict]) -> set[tuple]:
    """Chars printed over a different text run: horizontally overlapping, baselines less than half a line apart.
    Their letters interleave when read ("orersduinltasr" in 359_MATH_F211, where a table cell's text runs over the
    next cell's). Found on 10 of 540 handouts; normal text, kerning and fake bold never match."""
    buckets: dict[int, list[int]] = {}
    for i, c in enumerate(chars):
        buckets.setdefault(int(c["top"] // 4), []).append(i)
    bad = set()
    for key, indices in buckets.items():
        pool = sorted(indices + buckets.get(key + 1, []), key=lambda i: chars[i]["x0"])
        for n, i in enumerate(pool):
            a = chars[i]
            for j in pool[n + 1:]:
                b = chars[j]
                if b["x0"] >= a["x1"]:
                    break
                gap = abs(a["top"] - b["top"])
                width = min(a["x1"] - a["x0"], b["x1"] - b["x0"]) or 1
                if (0.5 < gap < 0.6 * min(a["size"], b["size"]) and a["text"].strip() and b["text"].strip()
                        and a["x1"] - b["x0"] > 0.3 * width):
                    bad |= {i, j}
    # why: the two runs only partly overlap, so their non-overlapping letters would stay behind as stray residue
    # ("r i ) e"); drop each run's whole stretch on its baseline, out to the nearest spaces
    spans: dict[float, list[float]] = {}
    for i in bad:
        span = spans.setdefault(round(chars[i]["top"], 1), [chars[i]["x0"], chars[i]["x1"]])
        span[0], span[1] = min(span[0], chars[i]["x0"]), max(span[1], chars[i]["x1"])
    for top, (lo, hi) in spans.items():
        line = sorted((i for i, c in enumerate(chars) if abs(c["top"] - top) < 0.3), key=lambda i: chars[i]["x0"])
        inside = [n for n, i in enumerate(line) if lo <= chars[i]["x0"] <= hi]
        if not inside:
            continue
        start, end = inside[0], inside[-1]
        while start > 0 and chars[line[start - 1]]["text"].strip():
            start -= 1
        while end < len(line) - 1 and chars[line[end + 1]]["text"].strip():
            end += 1
        bad.update(line[start:end + 1])
    return {(chars[i]["x0"], chars[i]["top"], chars[i]["text"]) for i in bad}


def without_overprint(page):
    """The page minus overprinted chars (see overprinted): losing that text beats storing it garbled."""
    bad = overprinted(page.chars)
    if not bad:
        return page
    return page.filter(lambda obj: obj.get("object_type") != "char" or (obj["x0"], obj["top"], obj["text"]) not in bad)


FOOTER = re.compile(r"(page \d+( of \d+)?|\d{1,2}|_+|please do not print unless necessary)", re.I)
TOP_ZONE, BOTTOM_ZONE = 4, 3  # lines per page where letterheads / footers sit


def drop_page_furniture(lines: list[Line]) -> list[Line]:
    """Remove running headers and footers, which otherwise glue onto whatever section spans a page break
    ("...gender-based violence. 1 BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, Pilani ...").

    A line is furniture if it sits in a page's top or bottom zone and either repeats in that zone on another
    page (digits ignored, so "Page 2 of 5" matches "Page 3 of 5"), or is a page number / print notice.
    """
    by_page: dict[int, list[int]] = {}
    for index, (page, _) in enumerate(lines):
        by_page.setdefault(page, []).append(index)
    zone = {i for indices in by_page.values() for i in indices[:TOP_ZONE] + indices[-BOTTOM_ZONE:]}
    key = lambda text: re.sub(r"\d+", "#", text.lower())  # noqa: E731
    # why per page, not per line: a header appears once per page; counting lines would double-count a page
    pages_with = Counter(key(text) for page, text in {(lines[i][0], lines[i][1]) for i in zone})
    return [
        line for i, line in enumerate(lines)
        if i not in zone or not (pages_with[key(line[1])] > 1 or FOOTER.fullmatch(line[1].strip()))
    ]


def cell_lines(cell: str | None) -> str:
    """Table cell with its line breaks kept (instructor cells list one name per line)."""
    return "\n".join(clean(part) for part in (cell or "").split("\n") if clean(part))


def load_manual() -> dict[str, Document]:
    """Hand-typed pages of scanned handouts ({} if the file is missing)."""
    if not MANUAL_PATH.exists():
        return {}
    docs = {}
    for record in json.loads(MANUAL_PATH.read_text())["documents"]:
        lines = [(page["page"], line) for page in record["pages"] for line in page["lines"]]
        tables = [(page["page"], table) for page in record["pages"] for table in page.get("tables", [])]
        docs[record["file"]] = Document(lines, tables)
    return docs


# ------------------------------ sections and header
def heading_of(line: str) -> tuple[str, str] | None:
    """(section, text after the colon) if the line starts a known section."""
    match = HEADING.match(line)
    if not match:
        return None
    text = match[1].lower().strip()
    return next(((key, match[2].strip()) for key, pattern in SECTIONS.items() if re.fullmatch(pattern, text)), None)


def split_sections(lines: list[Line]) -> dict[str, list[Line]]:
    """Lines grouped under the heading above them; the first occurrence of a section wins."""
    sections: dict[str, list[Line]] = {}
    current = None
    for page, line in lines:
        heading = heading_of(line)
        if heading:
            key, rest = heading
            current = key if key not in sections else None
            if current:
                sections[current] = [(page, rest)] if rest else []
        elif current:
            sections[current].append((page, line))
    return sections


def label_value(lines: list[Line], label: str, tables=()) -> tuple[str | None, int | None]:
    """Value after 'label :' in the first 40 lines, or next to a 'label' cell in a page-1 header table."""
    for page, rows in tables:
        for row in rows if page == 1 else []:
            cells = [c for c in row if c]
            if len(cells) >= 2 and re.fullmatch(rf"(?:{label})\W*", cells[0], re.I):
                return clean(cells[1]), page
    for page, line in lines[:40]:
        match = re.search(rf"\b(?:{label})\s*[:\-]\s*(.+)", line, re.I)
        if match and match[1].strip(" []"):
            return match[1].strip(" []"), page
    return None, None


INSTRUCTOR_LABEL = r"\W*(?:\d+\s*)?(?:team of |tutorial |lab |practical |course |lecture )?instructors?(?:[\s–-]*in[\s–-]*charge)?(?:\s*\([\w ]+\))?(?:\s*names?)?\W*"


NOT_A_NAME = r"^(na|n/a|tba|nil|none|name|-)$|time ?table|dean|agsr|supervisor|mentor|in-?charge|\bhod\b|building"


def extract_instructors(lines: list[Line], tables=()) -> tuple[list[str], int | None]:
    """Names on 'Instructor-in-charge : ...' lines, or next to an instructor label cell in a page-1 header table."""
    values = []  # (page, text holding names)
    for page, rows in tables:
        for row in rows if page == 1 else []:
            cells = [c for c in row if c]
            if len(cells) >= 2 and re.fullmatch(INSTRUCTOR_LABEL, cells[0], re.I):
                for cell in cells[1:]:  # one name per line; a one-word line is a wrapped surname
                    parts = cell.split("\n")
                    joined = [p for i, p in enumerate(parts) if i == 0 or len(p.split()) > 1 or "(" in p]
                    for i, part in enumerate(parts[1:], 1):
                        if len(part.split()) == 1 and "(" not in part:
                            joined[[j for j, q in enumerate(joined) if parts.index(q) < i][-1]] += f" {part}"
                    values.append((page, ", ".join(joined)))
    for page, line in lines[:40]:
        match = re.match(rf"^({INSTRUCTOR_LABEL}?)\s*[:\-–]?\s+(.+)", line, re.I)  # the colon is often missing
        if match and re.search(r"instructor", match[1], re.I):
            values.append((page, match[2]))
    names, page_found = [], min((page for page, _ in values), default=None)
    for page, text in values:
        value = re.sub(r"[\w.+-]+\s?@[\w.-]+|\([^)]*(?:\)|$)|\[|\]|https?://\S+|\b(?:lecture|tutorial|lab|practical)s?\s*:|\bI/?C\b", " ", text, flags=re.I)
        value = re.sub(rf"{INSTRUCTOR_LABEL}\s*:", ",", value, flags=re.I)  # "A Practical Instructors : B" on one line
        for name in re.split(r",|;|\band\b|&", value):
            name = clean(name).strip(" .:-")
            if name[:1].isupper() and len(name.split()) <= 5 and not re.search(r"\d|\b(is|are|the|of|for|to|course)\b", name, re.I) \
                    and not re.search(NOT_A_NAME, name, re.I):
                names.append(name)
    unique = {}  # case-insensitive: "SHASHI PRAKASH SINGH" and "Shashi Prakash Singh" are one person
    for name in names:
        unique.setdefault(re.sub(r"\W", "", name.lower()), name)
    return list(unique.values()), page_found


def title_matches(title: str | None, timetable_title: str | None) -> bool:
    """'Introduction to Biological Sciences' vs timetable 'INTRO TO BIO SCIENCES': timetable words prefix title words."""
    if not title or not timetable_title:
        return False
    words = re.findall(r"[a-z0-9]+", title.lower())
    wanted = [w for w in re.findall(r"[a-z0-9]+", timetable_title.lower()) if w not in ("and", "of", "the", "in", "to", "for")]
    return bool(wanted) and sum(any(x.startswith(w) for x in words) for w in wanted) >= 0.75 * len(wanted)


def build_about(sections: dict[str, list[Line]]) -> tuple[str | None, int | None]:
    """Description + objectives + outcomes, list markers and lead-in lines ('... will be able to:') removed."""
    pieces, page = [], None
    for key in ("description", "objectives", "outcomes"):
        for line_page, line in sections.get(key, []):
            page = page or line_page
            line = LIST_MARKER.sub("", line).strip()
            if line and not line.endswith(":"):
                pieces.append(line)
    text = re.sub(r"(\w)- (\w)", r"\1\2", " ".join(pieces))
    return text or None, page


def dehyphenate(text: str) -> str:
    return re.sub(r"(\w)- (\w)", r"\1\2", text)


def section_items(lines: list[Line]) -> list[str]:
    """A section as list items: a new item at each list marker, or one per sentence when there are no markers.
    Lead-ins ("Students will be able to:") are dropped, and each item starts with a capital."""
    # why: Word's Symbol-font bullets arrive as private-use glyphs (\uf0b7 bullet, \uf020 space), often several per line
    texts = []
    for _, line in lines:
        line = PRIVATE_BULLET.sub("• ", line.replace("\uf020", " "))
        parts = [("• " + part if n else part).strip() for n, part in enumerate(line.split("•")) if part.strip()]
        texts += [piece for part in parts for piece in INLINE_LETTER.split(part)]
    if sum(bool(LIST_MARKER.match(t)) for t in texts) >= 2:
        items: list[str] = []
        for text in texts:
            if LIST_MARKER.match(text) or not items:
                items.append(LIST_MARKER.sub("", text))
            else:
                items[-1] += " " + text
    else:
        items = [piece for sentence in SENTENCE_END.split(" ".join(texts)) for piece in INLINE_LETTER.split(sentence)]
    # a very long item is a paragraph under one bullet: one item per sentence reads better
    items = [piece for item in items for piece in (SENTENCE_END.split(item) if len(item) > 250 else [item])]
    result = []
    for item in items:
        item = LEAD_IN.sub("", dehyphenate(item).strip()).strip(" ;,")
        item = re.sub(r"^[A-Ha-h][.)]\s+", "", item)  # "A. Understand ...", "b. To engage ..." lettered items
        if len(re.findall(r"[A-Za-z]{3,}", item)) >= 3 and not item.endswith(":"):
            result.append(item[0].upper() + item[1:])
    return result


def about_parts(sections: dict[str, list[Line]]) -> dict:
    """The description as a paragraph, objectives and outcomes as lists (what build_about merges into one text)."""
    description = dehyphenate(" ".join(LIST_MARKER.sub("", PRIVATE_BULLET.sub("", line.replace("\uf020", " ")))
                                       for _, line in sections.get("description", []))).strip()
    return {"description": description or None, "objectives": section_items(sections.get("objectives", [])),
            "outcomes": section_items(sections.get("outcomes", []))}


# ------------------------------ tables: topics and evaluation
def is_plan_header(head: list[str]) -> bool:
    text = " ".join(head)
    return bool(re.search(r"topic|module|lecture|session|content|experiment", text)) and not re.search(r"weigh|marks", text)


def is_eval_header(head: list[str]) -> bool:
    text = " ".join(head)
    return bool(re.search(r"weigh|marks|%|percent", text) and re.search(r"component|evaluation|duration|date", text))


def table_groups(tables, header_test) -> list[tuple[list[list[str]], int]]:
    """Tables whose first row passes header_test, each merged with its continuations on later pages
    (same width, and either the same header again or no header of its own)."""
    groups, current = [], None
    for page, rows in tables:
        head = [clean(c).lower() for c in rows[0]]
        if current is not None and len(current) == 1 and len(rows[0]) == len(current[0]) + 1:
            current[0] = [""] + current[0]  # why: a header-only table whose empty first cell pdfplumber dropped
        repeated = current is not None and head == [clean(c).lower() for c in current[0]]
        if current and len(rows[0]) == len(current[0]) and (repeated or not (is_plan_header(head) or is_eval_header(head))):
            current.extend(rows[1:] if repeated else rows)
        elif header_test(head):
            current = list(rows)
            groups.append((current, page))
        else:
            current = None
    return groups


def column(head: list[str], pattern: str, skip=()) -> int | None:
    return next((i for i, name in enumerate(head) if i not in skip and re.search(pattern, name)), None)


def value_at(head: list[str], row: list[str], index: int | None) -> str:
    """The row's cell under a header; if empty, a neighbour whose header is blank (merged header cells)."""
    if index is None:
        return ""
    for i in (index, index - 1, index + 1):
        if 0 <= i < len(row) and row[i] and (i == index or (i < len(head) and not head[i])):
            return clean(row[i])
    return ""


def extract_topics(tables) -> tuple[list[str], int | None, bool, list[dict]]:
    """(topics, page, plan table found, groups): the module column + the best topic column of every plan table.
    groups keeps each module with the topics listed under it: [{"module": "Laplace transform", "topics": [...]}]."""
    topics, first_page, groups = [], None, []
    for rows, page in table_groups(tables, is_plan_header):
        head = [clean(c).lower() for c in rows[0]]
        numbers = tuple(i for i, n in enumerate(head) if re.search(r"\bno\.?$|number|^s\.? ?n", n))
        module = column(head, r"modul")  # "Module Number" columns hold the module names
        # why: "Lectures" may be a column of lecture numbers next to "Lecture session"; the wordiest candidate wins
        candidates = [i for i, n in enumerate(head) if i != module and re.search(r"topic|descri|content|session|lecture|experiment|title", n)]
        topic = max(candidates, key=lambda i: sum(len(re.findall(r"[A-Za-z]{3,}", value_at(head, r, i))) for r in rows[1:]), default=None)
        first_page = first_page or page
        for row in rows[1:]:
            for col in (module, topic):
                for part in re.split(r"[•▪●]", value_at(head, row, col)):
                    part = TOPIC_PREFIX.sub("", dehyphenate(part)).strip(" ,;:-–")
                    if not re.search(r"[A-Za-z]{3}", part):
                        continue
                    # why: a cell cut across rows (or pages) continues in lowercase: "with variable coefficients"
                    continues = part[0].islower()
                    if col == module and col is not None:
                        if continues and groups and groups[-1]["module"]:
                            groups[-1]["module"] += " " + part
                        else:
                            groups.append({"module": part, "topics": []})
                    elif continues and groups and groups[-1]["topics"]:
                        groups[-1]["topics"][-1] += " " + part
                    elif part.lower() not in {t.lower() for t in (groups[-1]["topics"] if groups else [])}:
                        if not groups:
                            groups.append({"module": None, "topics": []})
                        groups[-1]["topics"].append(part)
                    if part.lower() not in {t.lower() for t in topics}:
                        topics.append(part)
    return topics, first_page, first_page is not None, [g for g in groups if g["module"] or g["topics"]]


def parse_weight(text: str) -> float | None:
    """'25%' / '25' / '50 (25 %)' -> 25 (a percent beats marks)."""
    percent = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    number = percent or re.search(r"\d+(?:\.\d+)?", text)
    return float(number[1] if percent else number[0]) if number else None


def minutes(text: str) -> int | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(min|mins|minutes|hr|hrs|hours?|h)\b", text.lower())
    return round(float(match[1]) * (60 if match[2].startswith("h") else 1)) if match else None


def component(name: str, weight: float | None, duration: str, date: str, remarks: str) -> dict:
    name = re.sub(r"^[\uf000-\uf8ff•▪●◦➢*#\s]+", "", name).strip()  # PDF bullet glyphs before the name
    lowered = f"{name} {remarks}".lower()
    closed, opened = re.search(r"closed?[\s-]*book|\bcb\b", lowered), re.search(r"open[\s-]*book|\bob\b", lowered)
    group, individual = re.search(r"\bgroup|\bteam", lowered), re.search(r"individual", lowered)
    return {"name": name, "kind": next((k for k, p in KINDS.items() if re.search(p, name.lower())), "other"),
            "weightage_percent": weight, "duration_minutes": minutes(duration),
            "nature": None if bool(closed) == bool(opened) else ("CB" if closed else "OB"),
            "is_group": None if bool(group) == bool(individual) else bool(group),
            "date": date or None, "remarks": remarks or None}


def extract_evaluation(tables) -> tuple[list[dict], int | None, list[str], list[str]]:
    """(components, page, issues, notes) from the first evaluation table; marks are converted to percent."""
    groups = table_groups(tables, is_eval_header)
    if not groups:
        return [], None, ["evaluation_not_found"], []
    rows, page = groups[0]
    head = [clean(c).lower() for c in rows[0]]
    weight = column(head, r"weigh|%|percent") if column(head, r"weigh|%|percent") is not None else column(head, r"marks")
    name = column(head, r"component|evaluation|modules", (weight,))
    name = name if name is not None else next((i for i in range(len(head)) if i != weight), None)
    if name is None or weight is None:
        return [], page, ["evaluation_not_found"], []
    duration, date = column(head, r"duration", (name, weight)), column(head, r"date", (name, weight))
    remark = column(head, r"remark|comment|nature", (name, weight))
    cell = lambda row, i: value_at(head, row, i)
    components = []
    for row in rows[1:]:
        label = cell(row, name)
        if re.search(r"[A-Za-z]{3}", label) and not re.match(r"total", label, re.I):
            components.append(component(label, parse_weight(cell(row, weight)), cell(row, duration), cell(row, date), cell(row, remark)))
    issues, notes = [], []
    weights = [c["weightage_percent"] for c in components if c["weightage_percent"] is not None]
    total = sum(weights)
    if weights and "mark" in head[weight] and "%" not in head[weight] and "weigh" not in head[weight]:
        for c in components:
            if c["weightage_percent"] is not None:
                c["weightage_percent"] = round(c["weightage_percent"] * 100 / total, 2)
        notes.append("weightage_in_marks_converted")
        total = 100
    if not weights:
        issues.append("weightage_missing")
    elif abs(total - 100) > WEIGHT_TOLERANCE:
        issues.append(f"weightage_sum_{total:g}")
    return components, page, issues, notes


# ------------------------------ attendance and make-up (models)
def sentences_of(section: list[Line] | None, doc: Document, keyword: str):
    """Sentences of the policy section; without one, sentences anywhere in the handout containing the keyword."""
    from handout_models import Sentence
    out = []
    for page, line in section or doc.lines:
        for part in re.split(r"(?<=[.;])\s+", line):
            if len(part.split()) >= 3 and (section or re.search(keyword, part, re.I)):
                out.append(Sentence(part.strip(), page))
    return out


def decide(models, premises, hypotheses, field_name: str, issues: list[str]) -> tuple[bool | None, dict]:
    """(value, method entry). Confident entail/contradict -> True/False; confident neutral -> null; else null + issue."""
    decision = models.decide(premises, hypotheses)
    entry = {"method": "nli", "confidence": round(decision.confidence, 3) if decision.confidence else None,
             "evidence": decision.evidence.text if decision.evidence else None}
    if decision.confidence and decision.confidence >= NLI_MIN_CONFIDENCE:
        return decision.value, entry
    if decision.neutral_confidence < NLI_MIN_CONFIDENCE:
        issues.append(f"{field_name}_uncertain")
    return None, entry


NOT_MENTIONED = {"method": "nli", "confidence": None, "evidence": None}


def extract_attendance(models, doc, section, issues) -> tuple[dict, dict]:
    sentences = sentences_of(section, doc, r"attend")
    located = [item.sentence for item in models.locate(sentences, "regular attendance at lectures and classes", TOP_K)
               if item.similarity >= MIN_SIMILARITY]
    text = " ".join(line for _, line in section) if section else " ".join(s.text for s in located) or None
    percent = re.search(r"(\d{2,3})\s*%", text or "")
    if not located:
        return {"required": None, "percent": None, "text": text}, NOT_MENTIONED
    value, entry = decide(models, located, HYPOTHESES["attendance"], "attendance_required", issues)
    return {"required": value, "percent": int(percent[1]) if percent else None, "text": text}, entry


def extract_makeup(models, doc, section, kinds, issues) -> tuple[dict, dict]:
    """One decision per evaluation kind. why: a clause may decide a kind only if it names it; a clause naming
    no component is a general rule; "no make-up for quizzes" must never decide the midsem."""
    from handout_models import Sentence
    sentences = sentences_of(section, doc, MAKEUP_WORD)
    if not section:
        sentences = [item.sentence for item in models.locate(sentences, "make-up policy for a missed test", TOP_K)
                     if item.similarity >= MIN_SIMILARITY]
    clauses = [Sentence(part.strip(), s.page) for s in sentences for part in re.split(r";|\bhowever\b|\bbut\b", s.text, flags=re.I)
               if len(part.split()) >= 3 and re.search(MAKEUP_WORD, part, re.I)]
    per_component, methods = {}, {}
    for kind in kinds:
        naming = [c for c in clauses if kind in KINDS and re.search(KINDS[kind], c.text.lower())]
        general = [c for c in clauses if not any(re.search(p, c.text.lower()) for p in KINDS.values())]
        if naming:
            hypotheses = [(h.format(label=LABELS[kind]), yes, no) for h, yes, no in HYPOTHESES["makeup_component"]]
            per_component[kind], methods[kind] = decide(models, naming, hypotheses, f"makeup_{kind}", issues)
        elif general and kind != "class_participation":
            per_component[kind], methods[kind] = decide(models, general, HYPOTHESES["makeup_general"], f"makeup_{kind}", issues)
        else:
            per_component[kind], methods[kind] = None, NOT_MENTIONED
    stated = [v for v in per_component.values() if v is not None]
    allowed = True if True in (per_component.get("midsem"), per_component.get("compre")) else (False if stated and not any(stated) else None)
    return {"per_component": per_component, "allowed": allowed, "text": " ".join(s.text for s in sentences) or None}, methods


# ------------------------------ one handout
def printed_code_issue(name: str, course_no: str, printed: str | None, title: str | None, titles: dict[str, str]) -> tuple[str, bool] | None:
    """(tag, is_issue) when the printed course code differs from the file name. Only a different title as well
    means a different course; otherwise it is a cross-listing or old/new code pair (a note)."""
    codes = [f"{m[1]} {m[2]}{m[3]}" for m in re.finditer(r"\b([A-Z]{2,5})\s*([A-Z])\s?(\d{3}[A-Z]?)\b", (printed or "").upper())]
    if not codes or course_no in codes:
        return None
    raw_code = re.sub(r"^\d+_([A-Z]+)_", r"\1 ", Path(name).stem)  # the timetable keeps suffixes: "BITS F101-1"
    same = title_matches(title, titles.get(course_no) or titles.get(raw_code))
    return f"printed_code_{'differs' if same else 'mismatch'}: {'/'.join(codes)}", not same


def extract_handout(name: str, doc: Document, models, titles: dict[str, str], overrides: dict | None = None) -> Handout:
    match = re.match(r"\d+_([A-Z]+)_([A-Z]\d{3}[A-Z]?)", name)
    course_no = f"{match[1]} {match[2]}"
    issues, notes, overrides = [], [], overrides or {}
    sections = split_sections(doc.lines)

    title, title_page = label_value(doc.lines, r"course\s*(?:title|name)|title of the course|name of the course", doc.tables)
    if not title:
        issues.append("title_not_found")
    printed, printed_page = label_value(doc.lines, r"course\s*(?:no\.?|number|code)", doc.tables)
    if code := printed_code_issue(name, course_no, printed, title, titles):
        (issues if code[1] else notes).append(code[0])
    instructors, instructors_page = extract_instructors(doc.lines, doc.tables)
    about, about_page = build_about(sections)
    if not about:
        issues.append("about_not_found")

    topics, topics_page, plan_found, topic_groups = extract_topics(doc.tables)
    if plan_found and len(topics) < MIN_TOPICS:
        issues.append("topics_possibly_truncated")
    elif not topics:
        plan = " ".join(line for _, line in sections.get("plan", []))
        individual = re.search(r"project|thesis|dissertation|seminar|study|reading|practice", title or "", re.I) or "decided by" in plan
        (notes if individual else issues).append("no_course_plan_in_handout" if individual else "topics_not_found")
    evaluation, evaluation_page, eval_issues, eval_notes = extract_evaluation(doc.tables)
    issues, notes = issues + eval_issues, notes + eval_notes

    # Hand-checked values (build_handout_overrides.py) replace what the rules couldn't read, and clear their issues.
    override = overrides.get(name, {})
    fixed = {"evaluation": ("evaluation", "weightage"), "topics": ("topics",), "title": ("title",), "about": ("about",)}
    for key, prefixes in fixed.items():
        if key in override:
            issues = [i for i in issues if not i.startswith(prefixes)]
            notes = [n for n in notes if not (key == "topics" and n == "no_course_plan_in_handout") and
                     not (key == "evaluation" and n.startswith("weightage"))]
    evaluation = override.get("evaluation", evaluation)
    topics, title, about = override.get("topics", topics), override.get("title", title), override.get("about", about)
    if "topics" in override:
        topic_groups = [{"module": None, "topics": topics}] if topics else []
    notes += override.get("notes", []) + (["manually_checked"] if override else [])

    attendance, attendance_method = extract_attendance(models, doc, sections.get("attendance"), issues)
    kinds = list(dict.fromkeys(c["kind"] for c in evaluation))
    makeup, makeup_methods = extract_makeup(models, doc, sections.get("makeup"), kinds, issues)
    page_of = lambda key: sections[key][0][0] if sections.get(key) else None
    return Handout(
        file=name, course_no=course_no, title=title, instructors=instructors, about=about, topics=topics,
        evaluation=evaluation, attendance=attendance, makeup=makeup,
        consultation_hours=" ".join(line for _, line in sections.get("consultation", [])) or None,
        extraction_methods={"attendance.required": attendance_method, **{f"makeup.per_component.{k}": v for k, v in makeup_methods.items()}},
        source_pages={"course_no": printed_page, "title": title_page, "instructors": instructors_page, "about": about_page,
                      "topics": topics_page, "evaluation": evaluation_page, "attendance": page_of("attendance"),
                      "makeup": page_of("makeup"), "consultation_hours": page_of("consultation")},
        issues=issues, notes=notes, needs_verification=bool(issues),
        topic_groups=topic_groups, **about_parts(sections),
    )


# ------------------------------ run
def load_overrides() -> dict:
    try:
        return json.loads(OVERRIDES_PATH.read_text())
    except FileNotFoundError:
        return {}


def load_titles() -> dict[str, str]:
    try:
        return {c["course_no"]: c["title"] for c in json.loads(TIMETABLE_PATH.read_text())["courses"]}
    except (OSError, ValueError, KeyError) as error:
        log.warning("Timetable titles unavailable (%s); every printed-code difference is flagged", error)
        return {}


def summarise(records: list[Handout]) -> None:
    tag = lambda t: re.sub(r"(_\d.*|: .*)$", "", t)
    print(f"\nFiles: {len(records)}   flagged (needs_verification): {sum(r.needs_verification for r in records)}")
    print("Issues:", dict(Counter(tag(t) for r in records for t in r.issues).most_common()))
    print("Notes:", dict(Counter(tag(t) for r in records for t in r.notes).most_common()))
    counts = [len(r.topics) for r in records]
    print(f"Topics per handout: min {min(counts)}, median {statistics.median(counts)}, max {max(counts)}")
    print("attendance.required:", dict(Counter(r.attendance["required"] for r in records)))
    print("makeup.per_component:", dict(Counter(v for r in records for v in r.makeup["per_component"].values())))


def main(selected: list[str]) -> None:
    from handout_models import Models  # why: torch loads only when the models are actually needed
    files = [HANDOUTS_DIR / n for n in selected] if selected else sorted(HANDOUTS_DIR.glob("*.pdf"))
    if not files or not all(f.exists() for f in files):
        raise SystemExit(f"Handouts not found in {HANDOUTS_DIR}")
    docs = load_manual()
    to_read = [f for f in files if f.name not in docs]
    with multiprocessing.Pool(8) as pool:  # why: table detection is ~1 s per PDF; 8 processes cut the run to minutes
        docs.update(zip([f.name for f in to_read], pool.map(load_pdf, to_read)))
    models, titles = Models(EMBEDDING_MODEL, NLI_MODEL), load_titles()  # after the pool: don't fork torch state
    overrides = load_overrides()
    records = [extract_handout(f.name, docs[f.name], models, titles, overrides) for f in files if docs.get(f.name)]
    output = OUTPUT_PATH.with_name("handouts_sample.json") if selected else OUTPUT_PATH
    if not selected and OUTPUT_PATH.exists():
        OUTPUT_PATH.replace(OUTPUT_PATH.with_name("handouts_old.json"))
    output.write_text(json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False))
    log.info("Saved %d records to %s", len(records), output)
    summarise(records)


if __name__ == "__main__":
    main(sys.argv[1:])
