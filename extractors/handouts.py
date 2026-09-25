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
TIMETABLE_PATH = PROJECT_ROOT / "dataset" / "code processed" / "timetable.json"

SECTIONS = {  # section -> heading text (the part before ':'), matched at the start of a line
    "description": r"(course )?description",
    "objectives": r"(course )?(scope|objectives?|aims?)\b.*",
    "outcomes": r"(course )?(learning )?outcomes?",
    "plan": r"(course|lecture) plan.*|modules|plan of work|list of experiments",
    "evaluation": r"evaluation( scheme| schedule| components?)?",
    "attendance": r"attendance( policy)?",
    "makeup": r"make ?-?up( policy)?",
    "consultation": r"(chamber )?consultation.*",
    "other": r"text ?books?.*|reference.*|notices?.*|method of .*|nc .*|criterion.*|grading.*|note.*",
}
HEADING = re.compile(r"^(?:\d{1,2}(?:\.\d)?\s*[.):]?\s*)?([A-Za-z][A-Za-z &/-]{2,40}?)\s*(?::|$)(.*)$")
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
TOPIC_PREFIX = re.compile(r"^\s*((M|L|Lec|Lectures?|Module|Unit|Week)\.?\s*-?\s*\d+(\s*(-|–|to)\s*L?\d+)?\s*[:.)\-–]?\s*|\d{1,2}\s*[.)]\s*)", re.I)

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
    about: str | None
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


# ------------------------------ reading
def load_pdf(path: Path) -> Document | None:
    """Text lines and tables of every page; None (logged) if the PDF can't be read."""
    try:
        with pdfplumber.open(path) as pdf:
            pages = [page.dedupe_chars() for page in pdf.pages]  # dedupe: "fake bold" letters are drawn twice
            lines = [(p.page_number, clean(t)) for p in pages for t in (p.extract_text(x_tolerance=1.5) or "").splitlines() if clean(t)]
            tables = [(p.page_number, [[clean(c or "") for c in row] for row in t]) for p in pages for t in p.extract_tables() if t]
        return Document(lines, tables)
    except Exception:
        log.exception("Could not read %s", path.name)
        return None


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


def label_value(lines: list[Line], label: str) -> tuple[str | None, int | None]:
    """Value after 'label :' in the first 40 lines (the header block)."""
    for page, line in lines[:40]:
        match = re.search(rf"\b(?:{label})\s*[:\-]\s*(.+)", line, re.I)
        if match and match[1].strip(" []"):
            return match[1].strip(" []"), page
    return None, None


def extract_instructors(lines: list[Line]) -> tuple[list[str], int | None]:
    """Names on 'Instructor-in-charge : ...' / 'Instructors : ...' lines, IC first."""
    names, page_found = [], None
    for page, line in lines[:40]:
        match = re.match(r"^\W*((?:team of |tutorial |lab |practical |course )?instructors?(?:[\s–-]*in[\s–-]*charge)?)\s*[:\-–]\s*(.+)", line, re.I)
        if not match:
            continue
        page_found = page_found or page
        value = re.sub(r"[\w.+-]+@[\w.-]+|\(.*?\)|\[|\]|https?://\S+", " ", match[2])
        for name in re.split(r",|;|\band\b|&", value):
            name = clean(name).strip(" .:-")
            if name and len(name.split()) <= 5 and not re.search(r"\d|\b(is|are|the|of|for|to|course)\b", name, re.I) \
                    and name.lower() not in ("na", "n/a", "tba", "nil"):
                names.append(name)
    return list(dict.fromkeys(names)), page_found


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
        head = [c.lower() for c in rows[0]]
        repeated = current is not None and head == [c.lower() for c in current[0]]
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
            return row[i]
    return ""


def extract_topics(tables) -> tuple[list[str], int | None, bool]:
    """(topics, page, plan table found): the module column + the best topic column of every plan table."""
    topics, first_page = [], None
    for rows, page in table_groups(tables, is_plan_header):
        head = [c.lower() for c in rows[0]]
        numbers = tuple(i for i, n in enumerate(head) if re.search(r"\bno\.?$|number|^s\.? ?n", n))
        module = column(head, r"modul")  # "Module Number" columns hold the module names
        topic = next((column(head, p, (module, *numbers)) for p in (r"topic", r"descri|content", r"session|lecture", r"experiment|title")
                      if column(head, p, (module, *numbers)) is not None), None)
        first_page = first_page or page
        for row in rows[1:]:
            for col in (module, topic):
                for part in re.split(r"[•▪●]", value_at(head, row, col)):
                    part = TOPIC_PREFIX.sub("", part).strip(" ,;:-–")
                    if re.search(r"[A-Za-z]{3}", part) and part.lower() not in {t.lower() for t in topics}:
                        topics.append(part)
    return topics, first_page, first_page is not None


def parse_weight(text: str) -> float | None:
    """'25%' / '25' / '50 (25 %)' -> 25 (a percent beats marks)."""
    percent = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    number = percent or re.search(r"\d+(?:\.\d+)?", text)
    return float(number[1] if percent else number[0]) if number else None


def minutes(text: str) -> int | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(min|mins|minutes|hr|hrs|hours?|h)\b", text.lower())
    return round(float(match[1]) * (60 if match[2].startswith("h") else 1)) if match else None


def component(name: str, weight: float | None, duration: str, date: str, remarks: str) -> dict:
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
    head = [c.lower() for c in rows[0]]
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


def extract_handout(name: str, doc: Document, models, titles: dict[str, str]) -> Handout:
    match = re.match(r"\d+_([A-Z]+)_([A-Z]\d{3}[A-Z]?)", name)
    course_no = f"{match[1]} {match[2]}"
    issues, notes = [], []
    sections = split_sections(doc.lines)

    title, title_page = label_value(doc.lines, r"course\s*(?:title|name)|title of the course|name of the course")
    if not title:
        issues.append("title_not_found")
    printed, printed_page = label_value(doc.lines, r"course\s*(?:no\.?|number|code)")
    if code := printed_code_issue(name, course_no, printed, title, titles):
        (issues if code[1] else notes).append(code[0])
    instructors, instructors_page = extract_instructors(doc.lines)
    about, about_page = build_about(sections)
    if not about:
        issues.append("about_not_found")

    topics, topics_page, plan_found = extract_topics(doc.tables)
    if plan_found and len(topics) < MIN_TOPICS:
        issues.append("topics_possibly_truncated")
    elif not topics:
        plan = " ".join(line for _, line in sections.get("plan", []))
        individual = re.search(r"project|thesis|dissertation|seminar|study|reading|practice", title or "", re.I) or "decided by" in plan
        (notes if individual else issues).append("no_course_plan_in_handout" if individual else "topics_not_found")
    evaluation, evaluation_page, eval_issues, eval_notes = extract_evaluation(doc.tables)
    issues, notes = issues + eval_issues, notes + eval_notes

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
    )


# ------------------------------ run
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
    records = [extract_handout(f.name, docs[f.name], models, titles) for f in files if docs.get(f.name)]
    output = OUTPUT_PATH.with_name("handouts_sample.json") if selected else OUTPUT_PATH
    if not selected and OUTPUT_PATH.exists():
        OUTPUT_PATH.replace(OUTPUT_PATH.with_name("handouts_old.json"))
    output.write_text(json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False))
    log.info("Saved %d records to %s", len(records), output)
    summarise(records)


if __name__ == "__main__":
    main(sys.argv[1:])
