"""Course handout extractor: dataset/raw/handouts/*.pdf -> dataset/code processed/handouts.json.

Usage:
    .venv/bin/python extractors/handouts.py                    # all handouts (renames old output first)
    .venv/bin/python extractors/handouts.py 001_AN_F314.pdf …  # only these files -> handouts_sample.json
"""
import copy
import json
import logging
import multiprocessing
import os
import re
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber

from timetable import PROJECT_ROOT, clean

# =====================================================================================
# CONFIG: extraction method per field (one method per field, the same for every handout)
# =====================================================================================
FIELD_METHODS = {
    "file": ("rule", "the PDF's file name"),
    "course_no": ("rule", "fixed 'NNN_DEPT_CODE.pdf' file-name format"),
    "title": ("rule", "always a labelled header line (Course Title / Name of the course / ...)"),
    "instructors": ("rule", "labelled header lines (Instructor-in-charge / Team of instructors / ...)"),
    "about": ("rule", "sections found by heading keyword; lead-in lines dropped by list structure, "
                      "with an embedding similarity check as backup (the text itself is never generated)"),
    "topics": ("rule", "course plan is a table; columns found by header name"),
    "evaluation": ("rule", "evaluation scheme is a table (text fallback for unruled tables); values are structured"),
    "attendance.required": ("model", "wording varies a lot ('must have 80%', 'expected to attend', 'attendance is not "
                                     "mandatory'); needs meaning, incl. negation. Confident 'neutral' -> null"),
    "attendance.percent": ("rule", "a stated percentage like '75%' next to a minimum-attendance phrase"),
    "attendance.text": ("model", "the sentences the model located as being about attendance"),
    "makeup.per_component": ("model", "per-component exceptions and negations ('no make-up for quizzes; others in "
                                      "genuine cases') need sentence meaning. One rule inside: an 'only for <X>' "
                                      "clause excludes the components it doesn't name (NLI can't do that exclusion)"),
    "makeup.allowed": ("rule", "derived in code from per_component, never decided separately"),
    "makeup.text": ("model", "the sentences the model located as being about make-up"),
    "consultation_hours": ("rule", "section found by heading keyword, kept as written"),
    "source_pages": ("rule", "page of the line / table / evidence sentence each value came from"),
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
TOP_K_SENTENCES = 3        # candidate sentences passed from "locate" to "decide"
# Below this even the best sentence is "not mentioned" (null, no issue tag). Per field: MiniLM scores the
# boilerplate "In addition to Part I (General Handout ...)" around 0.45-0.50 against any policy query.
MIN_SIMILARITY = {"attendance": 0.45, "makeup": 0.40}
# Only when a handout has no section for the field: the whole-handout pool is narrowed to sentences containing
# the field's word stem before embedding. why: the small NLI model gives confident nonsense on off-topic
# sentences ("Detailed syllabus and lecture plan follows." -> entailment 0.96 for "required to attend").
FALLBACK_KEYWORDS = {"attendance": r"attend", "makeup": r"make\s*-?\s*ups?"}
NLI_MIN_CONFIDENCE = 0.70  # below this the decision is null + "<field>_uncertain"
HEADER_SIMILARITY = 0.72   # "about" lines at least this close to a header example are dropped (backup check)
HEADER_MAX_WORDS = 25      # only short lines can be headers
WRAPPED_LINE_CHARS = 70   # a line at least this long without end punctuation continues on the next line
TOPICS_MIN_EXPECTED = 3    # fewer topics than this from a plan table -> "topics_possibly_truncated" (4-5 big modules is normal)
TITLE_MATCH_SHARE = 0.75   # share of timetable title words that must match the handout title (printed code check)
INDIVIDUAL_COURSE_TITLE = r"project|thesis|dissertation|seminar|study in advance|reading course|case studies|research practice|practice school"
NO_PLAN_WORDING = r"decided by|will be decided|see detailed course schedule|as per the supervisor|individual"
WEIGHT_TOLERANCE = 1       # weightages must sum to 100 +- this
MIN_TEXT_COMPONENTS = 3    # a text-parsed scheme that doesn't add up to 100 is kept (flagged) only if this long
PDF_WORKERS = 8            # parallel PDF parsing processes (capped at the CPU count)
HEADING_MAX_WORDS = 6      # an unnumbered heading line without a colon has at most this many words
HEADER_BLOCK_LINES = 45    # the label: value header block is within the first lines of a handout
HEADER_MAX_ROWS = 4        # a table header can span this many rows

FIELD_DESCRIPTIONS = {
    "attendance": "regular attendance at lectures and classes",
    "makeup": "make-up policy of the course, whether a make-up test is allowed for a missed evaluation",
    "makeup_component": "make-up policy for the {label}",
}
# (hypothesis, value if entailed, value if contradicted); None = that label proves nothing for this hypothesis.
# why negations: the small model is far more decisive on negations of hedged make-up text ("No make-up
# except in case of hospitalization" -> contradicts "Make-up is never allowed."). Attendance uses entailment
# only: "Students are required to attend classes." is contradicted by "There is no marks for attendance",
# and "Attendance is optional." by "Attendance may be taken randomly"; neither means the opposite.
HYPOTHESES = {
    "attendance.required": [("Students are required to attend classes.", True, None), ("A minimum attendance is required.", True, None),
                            ("Students should attend lectures regularly.", True, None), ("Attendance is optional.", False, None)],
    # Tried first: only a clause that names the component can decide it.
    "makeup.component": [("A make-up is allowed for the {label}.", True, False), ("Make-up exams are possible for the {label}.", True, False),
                         ("There is no make-up for the {label}.", False, True)],
    # For clauses about "other components", used only for kinds that no clause names.
    "makeup.other_components": [("There is a make-up for other components.", True, False)],
    # Last, only on clauses that name no evaluation component ("Make-up will be granted only in genuine cases");
    # otherwise "no make-up for quizzes" would decide the midsem too.
    "makeup.general": [("Make-ups can be given.", True, False), ("Make-up is never allowed.", False, True)],
}
# A make-up clause restricted to named components ("make-up only for the midsem and compre"): components it
# doesn't name get False by rule. why: NLI can't do this exclusion ("Quizzes have a make-up." -> entailment 0.96).
MAKEUP_ONLY_FOR = r"\bonly\s+(?:(?:be\s+)?(?:for|to|in|applicable\s+to)\s+)?(?:the\s+)?(?:{0})|\bfor\s+only\s+(?:the\s+)?(?:{0})|\((?:[^)]*?(?:{0}))[^)]*\bonly\)"
MAKEUP_WORD = r"make\s*-?\s*ups?"
OTHER_COMPONENTS = r"\bother (?:evaluation )?(?:components?|evaluatives?|evaluations?|tests?)|\brest of\b"
# Clause boundaries inside one sentence ("No make-up for quizzes; however, for other components ...").
CLAUSE_SPLIT = r";|\bhowever\b,?|\bbut\b|\bwhereas\b|^while\b[^,]{3,80},"
KIND_LABELS = {
    "quiz": "quizzes", "midsem": "mid-semester exam", "compre": "comprehensive exam", "assignment": "assignments",
    "project": "project", "lab": "lab components", "tutorial_test": "tutorial tests", "presentation": "presentations",
    "viva": "viva", "seminar": "seminars", "class_participation": "class participation", "other": "other components",
}
HEADER_EXAMPLES = [
    "Following are the scope and objective of this course:",
    "Upon successful completion of this course, students will be able to:",
    "The students will have the following learning outcomes:",
    "At the end of the course, the student should be able to",
    "The objectives of this course are as follows:",
    "After completing this course the student will be able to:",
]
# =====================================================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
for noisy in ("httpx", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

HANDOUTS_DIR = PROJECT_ROOT / "dataset" / "raw" / "handouts"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "code processed" / "handouts.json"
OLD_OUTPUT_PATH = OUTPUT_PATH.with_name("handouts_old.json")
SAMPLE_OUTPUT_PATH = OUTPUT_PATH.with_name("handouts_sample.json")
# Hand-typed page text/tables for handouts with no text layer (scanned); fed through the same pipeline.
MANUAL_PATH = PROJECT_ROOT / "dataset" / "manually processed" / "handouts.json"
TIMETABLE_PATH = PROJECT_ROOT / "dataset" / "code processed" / "timetable.json"  # titles for the printed-code check

# why: the default (3) glues words together in PDFs with tight letter spacing ("additiontopartI").
X_TOLERANCE = 1.5

# Section key -> regex that must match the WHOLE heading text (the part before ':', or the whole line).
# Sections are found by keyword, never by number (Evaluation is section 5, 6 or 7 in different handouts).
SECTION_HEADINGS = {
    "description": r"((general|catalog|course) )*description( of the course)?",
    "objectives": r"(course )?(scope|objectives?)( ?(and|&) ?(objectives?|scope))?( of the course)?|aims?( and (learning )?objectives?)?",
    "learning_outcomes": r"(course )?(learning )?outcomes?",
    "textbooks": r"text ?(and|&) ?reference ?books?|text ?bo[o0]ks?( ?(and|&) ?(reference books?|references?))?",
    "references": r"references?( books?)?|reference books?|other references|lab(oratory)? manual",
    "course_plan": r"(course|lecture|lesson) ?plan|plan of work|lecture schedule|course (content|schedule)|(list of )?experiments?( list)?|modules",
    "evaluation": r"evaluation( scheme| schedule| components?| pattern| plan)?",
    "attendance": r"attendance( policy| requirements?)?",
    "makeup": r"make ?-?up( policy)?",
    "consultation": r".*consultation.*|office hours?",
}
# "5. Evaluation", "1: Course Description", "1. 2. Scope and Objective" (a doubled number), "2. a) Objective".
NUMBERED_HEADING = re.compile(r"(?:(?:\d{1,2}|[IVX]{1,4})\s*[.):]\s*|\d{1,2}\.\d{1,2}\s+)+(?:[a-z]\)\s*)?([A-Za-z].*)")  # also "4.1 Textbook:"
PAGE_NUMBER = re.compile(r"page \d+( of \d+)?", re.I)

NUMBER_LABEL = r"course\s*(?:no\.?|number|code)"
TITLE_LABEL = r"course\s*(?:no\.?|number)\s*&\s*title|course\s*(?:title|name)|name\s*of\s*the\s*course|title\s*of\s*the\s*course"
LABEL_LINE = re.compile(r"^\s*(?:\d+\.?\s*)?(?:[a-z]\)\s*)?([A-Za-z][A-Za-z /&\-–.]{1,45}?(?:\([\w /-]{1,8}\))?)\s*:\s*(.*)$")  # "Lecture Instructor(L2) :"
# An instructor label printed without a colon (header blocks laid out as tables): "Instructor-in-Charge Prof. X",
# "Instructors (tutorial) A B", "[Team of Instructors] A, B".
INSTRUCTOR_LINE = re.compile(r"^\s*(?:\d+\.?\s*)?\[?((?:course\s+)?(?:team\s+of\s+)?(?:tutorial\s+|practical\s+|lab\s+)?instructors?"
                             r"(?:\s*[-–]?\s*in\s*[-–]?\s*charge)?(?:\s*\((?:s|lec|tutorial|lab)\))?)\]?\s*[:\-–]?\s+(.+)$", re.I)
INSTRUCTOR_LABEL = re.compile(r"instructor|in[\s-]*charge|faculty|co-?ordinator|tutor|teacher|team", re.I)
NOT_A_NAME = re.compile(r"^(n/?a|tba|nil|none|-|–|ic|i/c|rs)$|\b(building|block|room|chamber|department|campus|pilani)\b", re.I)
HONORIFIC = re.compile(r"^(prof|dr|mr|ms|mrs|shri|smt)\.?\s+", re.I)
# A line with these words is a sentence, not a list of names ("The list of PhD lab instructors is provided below").
SENTENCE_WORDS = re.compile(r"\b(is|are|was|will|be|the|this|with|below|above|of|for|to|in|course|students?)\b", re.I)
MAX_NAME_WORDS = 5

# List markers at the start of an item: "a.", "(b)", "1.", "iv)", "CLO1.", "CO2:", bullets.
LIST_MARKER = re.compile(r"^\s*(\(?[a-zA-Z]\)|[a-zA-Z]\.(?=\s)|\(?\d{1,2}[.)](?!\d)|\(?[ivxIVX]{1,4}[.)]|CLO\s*\d+\s*[.:)]?|CO\s*\d+\s*[.:)]|"
                         r"[•▪●○■◦➢✓\-*])\s*")
BULLETS = re.compile(r"[•▪●○■◦➢✓]")

# Evaluation component name -> kind. First match wins, so "Lab Viva" is a viva and "Tutorial Quiz" a tutorial test.
# why \b: "tut" is inside "institute", "lab" inside "available"/"syllabus", "mid" inside "amid".
COMPONENT_KINDS = {
    "midsem": r"\bmid",
    "compre": r"\bcompre|\bfinal exam|\bend ?-?sem",
    "viva": r"\bviva",
    "tutorial_test": r"\btut",
    "quiz": r"\bquiz|\bclass tests?|\bsurprise tests?",
    "lab": r"\blab|\bpractical|\bexperiment",
    "assignment": r"\bassign|\bhome ?work|\btake[\s-]?home",
    "project": r"\bproject",
    "seminar": r"\bseminar",
    "presentation": r"\bpresentation",
    "class_participation": r"\bparticipation|\battendance",
}
HEADER_WORDS = r"topic|modul|lec|week|session|content|chapter|descri|experiment|practical|reference|outcome|objective|" \
               r"component|duration|date|weigh|marks|remark|nature|comment|\bno\b|number|book|hours"
EVALUATION_HEADER = re.compile(r"weigh|marks|%")
PLAN_HEADER = re.compile(r"topic|modul|\blec|week|\bwk\b|session|content|chapter|descriptor|description|coverage|experiment|practical")
MODULE_COLUMN = r"^mod|module"
TOPIC_COLUMNS = [r"topic", r"descriptor|description|details|content|coverage", r"lecture session|session|lecture plan|lecture obj",
                 r"experiment|title|practical"]
OBJECTIVE_COLUMN = r"learning obj|learning outcome"  # topics only when a plan table has nothing better
NUMBER_COLUMN = r"\bno\.?$|\bnumber$|^#$|^s\.? ?n"  # "Session No." holds numbers, never topics
# "M1:", "L1-3:", "L-1-2:", "L.1.1", "Lecture 4 to 6 -", "Unit 3", "1.", "3-6.", "3 Introduction" at the start of a topic.
TOPIC_PREFIX = re.compile(r"^\s*(?:(?:M|L|Lectures?|Lect|Lec|Modules?|Units?|Weeks?|W|Ch)\.?\s*-?\s*\d+(?:\.\d+)*"
                          r"(?:\s*(?:[-–]|to)\s*(?:L|M)?\.?\s*\d+(?:\.\d+)*)?\s*[:.)\-–]*\s*"
                          r"|\d{1,2}(?:\s*[-–]\s*\d{1,2})?\s*[.)]\s*|\d{1,2}\s+(?=[A-Z]))", re.I)
# A lecture prefix inside a cell starts a new topic ("... L3-L4: Agents of ...", "... L.5-L.6. DNA ...").
INNER_LECTURE = r"(?=\bL\.?\s*\d+(?:\.\d+)*\s*(?:[-–]\s*L?\.?\s*\d+(?:\.\d+)*)?\s*(?::|\.\s))"

Line = tuple[int, str]  # (page number, text)
Row = list[str | None]
PageTable = tuple[int, list[Row]]


@dataclass
class Document:
    """Everything the pipeline reads from one handout: text lines and tables, with page numbers."""

    lines: list[Line]          # text outside tables (sections are read from these)
    tables: list[PageTable]
    header_lines: list[Line]   # full page text incl. tables (some handouts put the header block in a table)


@dataclass
class Component:
    name: str
    kind: str
    weightage_percent: float | None
    duration_minutes: int | None
    nature: str | None  # "CB", "OB", or None (not stated, or both)
    is_group: bool | None
    date: str | None
    remarks: str | None


@dataclass
class Handout:
    file: str
    course_no: str
    title: str | None
    instructors: list[str]
    about: str | None
    topics: list[str]
    evaluation: list[Component]
    attendance: dict
    makeup: dict
    consultation_hours: str | None
    extraction_methods: dict = field(default_factory=dict)
    source_pages: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    needs_verification: bool = False


# ---------------------------------------------------------------- reading

def page_lines(pages) -> list[Line]:
    """All text lines with their page number, minus repeated page headers/footers.

    why: every page repeats the institute letterhead; a line in the top 4 or bottom 3 lines of at
    least two pages is boilerplate. dedupe_chars() undoes "fake bold" (letters drawn twice).
    """
    pages = [(page.page_number, (page.dedupe_chars().extract_text(x_tolerance=X_TOLERANCE) or "").splitlines()) for page in pages]
    edge_counts: Counter[str] = Counter()
    for _, lines in pages:
        edge_counts.update(set(lines[:4] + lines[-3:]))
    boilerplate = {line for line, count in edge_counts.items() if count >= 2}
    return [(number, clean(line)) for number, lines in pages for line in lines
            if clean(line) and line not in boilerplate and not PAGE_NUMBER.fullmatch(clean(line))]


def outside_tables(page):
    """The page without the characters inside detected tables.

    why: table cells otherwise leak into the text lines, and a cell like "Learning Outcomes"
    in a course-plan header gets mistaken for a section heading.
    """
    boxes = [table.bbox for table in page.find_tables({"text_x_tolerance": X_TOLERANCE})]
    inside = lambda obj, box: box[0] <= obj.get("x0", -1) and obj.get("x1", 1e9) <= box[2] and box[1] <= obj.get("top", -1) and obj.get("bottom", 1e9) <= box[3]
    return page.filter(lambda obj: not any(inside(obj, box) for box in boxes))


def load_pdf(pdf_file: Path) -> Document:
    with pdfplumber.open(pdf_file) as pdf:
        header_lines = page_lines(pdf.pages)
        lines = page_lines([outside_tables(page.dedupe_chars()) for page in pdf.pages])
        tables = [(page.page_number, table) for page in pdf.pages
                  for table in page.dedupe_chars().extract_tables({"text_x_tolerance": X_TOLERANCE}) if table]
    return Document(lines, tables, header_lines)


def load_manual_documents() -> dict[str, Document]:
    """file name -> Document typed in by hand (manually processed/handouts.json); {} if absent."""
    try:
        records = json.loads(MANUAL_PATH.read_text())["documents"]
    except FileNotFoundError:
        return {}
    except (ValueError, KeyError, TypeError) as error:
        log.warning("Could not read %s (%s); manual documents ignored", MANUAL_PATH, error)
        return {}
    documents = {}
    for record in records:
        lines = [(page["page"], line) for page in record["pages"] for line in page["lines"]]
        tables = [(page["page"], table) for page in record["pages"] for table in page.get("tables", [])]
        documents[record["file"]] = Document(lines, tables, lines)
    return documents


# ---------------------------------------------------------------- sections and header

def match_heading(line: str) -> tuple[str | None, str] | None:
    """(section key, or None for an unstored heading; text after the colon), or None if not a heading."""
    numbered = NUMBERED_HEADING.fullmatch(line)
    if not numbered and not (line[:1].isupper() or line[:1].isdigit()):
        return None  # why: "... regarding the course description." wraps onto a line that looks like a heading
    text = numbered[1] if numbered else line
    heading, colon, rest = text.partition(":")
    heading = clean(re.sub(r"\(.*?\)", "", heading)).rstrip(" .-–").lower()
    # why: an unnumbered line without a colon is a heading only if short; "Seminar III 25 To be announced by
    # the DCA in consultation with" is a table row, not a "Consultation" heading.
    if not numbered and not colon and len(heading.split()) > HEADING_MAX_WORDS:
        return None
    for key, pattern in SECTION_HEADINGS.items():
        if re.fullmatch(pattern, heading):
            return key, rest.strip()
    # why: an unknown numbered heading ("8. Notices:") still closes the previous section; a numbered row with
    # numbers in it ("1. Project Outline & Plan of Work 5 2nd week") is a list item, not a heading.
    if numbered and len(heading) <= 50 and (colon or not heading.endswith(".")) and not re.search(r"\d", heading):
        return None, ""
    # Edge case (e.g. 091_CE_G527): "1 Course Description:" is numbered without a dot; only known headings.
    bare = re.fullmatch(r"\d{1,2}\s+([A-Za-z].*)", line)
    if bare:
        heading, _, rest = bare[1].partition(":")
        heading = clean(re.sub(r"\(.*?\)", "", heading)).rstrip(" .-–").lower()
        for key, pattern in SECTION_HEADINGS.items():
            if re.fullmatch(pattern, heading):
                return key, rest.strip()
    return None


def split_sections(lines: list[Line]) -> dict[str, list[Line]]:
    """Group lines under the heading above them; the first occurrence of each section wins."""
    sections: dict[str, list[Line]] = {}
    current: str | None = None
    for page, line in lines:
        heading = match_heading(line)
        if heading is None:
            if current:
                sections[current].append((page, line))
            continue
        key, rest = heading
        current = None if key is None or key in sections else key
        if current:
            sections[current] = [(page, rest)] if rest else []
    return {key: body for key, body in sections.items() if body}


def header_block(lines: list[Line]) -> list[Line]:
    """Lines before the first known section heading after the course-number line (the label: value block).

    why: a few handouts open with a "Course Content" paragraph above the header block (286_ENVS_F212).
    """
    first = lines[:HEADER_BLOCK_LINES]
    number_at = next((i for i, (_, line) in enumerate(first) if re.search(NUMBER_LABEL, line, re.I)), 0)
    block: list[Line] = list(first[:number_at])
    for page, line in first[number_at:]:
        heading = match_heading(line)
        if heading and heading[0]:
            break
        block.append((page, line))
    return block


def header_value(block: list[Line], label: str) -> tuple[str | None, int | None]:
    """(value after 'label :', page) in the header block."""
    for page, line in block:
        # why: search, not match: some put a prefix first ("1. a) Course Number: CS F446").
        match = re.search(rf"\b(?:{label})\s*:?\s*(.+)", line, re.I)
        if match and match[1].strip(" []"):
            return match[1].strip(" []"), page
    return None, None


def course_no_from_filename(pdf_file: Path) -> str | None:
    """'002_BIO_F101.pdf' -> 'BIO F101'; suffixes like '-1' are stripped ('BITS F101-1' -> 'BITS F101')."""
    match = re.fullmatch(r"\d+_([A-Z]+)_([A-Z]\d{3}[A-Z]?)(-\d+)?", pdf_file.stem)
    return f"{match[1]} {match[2]}" if match else None


def printed_codes(printed: str) -> list[str]:
    """Course codes in a printed course-number line, tolerant of missing spaces.

    'GSF213' -> ['GS F213']; 'EEE / ECE / INSTR / CS F342' -> 4 codes; 'CS/SS G 527' -> ['CS G527', 'SS G527'].
    """
    printed = printed.upper()
    codes: list[str] = []
    for shared in re.finditer(r"((?:[A-Z]{2,5}\s*/\s*)+[A-Z]{2,5})\s+([A-Z])\s?(\d{3}[A-Z]?)", printed):
        codes += [f"{department.strip()} {shared[2]}{shared[3]}" for department in shared[1].split("/")]
    codes += [f"{m[1]} {m[2]}{m[3]}" for m in re.finditer(r"\b([A-Z]{2,5}?)\s*([A-Z])\s?(\d{3}[A-Z]?)\b", printed)]
    return list(dict.fromkeys(codes))


def extract_title(block: list[Line]) -> tuple[str | None, int | None]:
    title, page = header_value(block, TITLE_LABEL)
    printed, printed_page = header_value(block, NUMBER_LABEL)
    if title and printed and not printed_codes(printed) and printed_codes(title):
        title, page = printed, printed_page  # the two labels are swapped in the handout (269_EEE_G554)
    if title:  # combined "Course Number & Title : CS/SS G 527 Cloud Computing"
        title = re.sub(r"^(?:[A-Z]{2,5}\s*/\s*)*[A-Z]{2,5}\s*[A-Z]\s?\d{3}[A-Z]?\s*", "", title).strip() or None
    return title, page


def split_names(value: str) -> list[tuple[str, bool]]:
    """'Prof. A (IC), B and C (RS)' -> [('Prof. A', True), ('B', False), ('C', False)]. True = marked IC."""
    # why: brackets are often template brackets around the names ("[Tufan Chandra Bera]"), so only the
    # contact details inside them go: e-mails, URLs, "office 2220-M"; "[IC]" is an in-charge marker.
    value = re.sub(r"\[\s*(IC|I/C)\s*\]", "(IC)", value, flags=re.I)
    value = re.sub(r"[\w.+-]+@[\w.-]*\w|https?://[^\s\];,]+|\boffice\s+\S+", " ", value, flags=re.I)
    value = value.replace("[", " ").replace("]", " ")
    value = re.sub(r"\b(?:lecture|tutorials?|labs?|practicals?)\s*:", ",", value, flags=re.I)  # "Lecture: A, Tutorial: B"
    # why: remove "(...)" notes before splitting on commas ("A (IC, Room 2), B" must not split inside);
    # an IC note becomes a marker so that name can go first.
    value = re.sub(r"\(([^)]*)\)?", lambda m: " ICMARK " if re.search(r"\bIC\b|I/C|in[\s-]*charge", m[1], re.I) else " ", value)
    names: list[tuple[str, bool]] = []
    for part in re.split(r",|;|\band\b|&|\+", value):
        marked_ic = "ICMARK" in part
        name = clean(re.sub(r"ICMARK|email\s*:?|[\[\]]|\b(?:teaching assistants?|instructors?|tutors?)\b", " ", part, flags=re.I)).strip(" .:-)")
        if name and re.search(r"[A-Za-z]{2}", name) and not re.search(r"\d", name) and not NOT_A_NAME.search(name) \
                and len(name.split()) <= MAX_NAME_WORDS and not SENTENCE_WORDS.search(HONORIFIC.sub("", name)) \
                and not (match_heading(name) or (None,))[0]:
            names.append((name, marked_ic))
    return names


def instructor_groups_from_tables(tables: list[PageTable]) -> tuple[list[list], int | None]:
    """[is_ic_label, text] groups from header tables: a label cell ("Instructors (lab)") followed by value cells
    with one name per line. why: flattened to text, two value columns merge pairs of names ("Rejaul Raja Piyush")."""
    groups: list[list] = []
    page_found: int | None = None
    for page, table in tables:
        if page > 2:
            break
        for row in table:
            cells = [cell or "" for cell in row]
            for index, cell in enumerate(cells):
                label = cell_text(cell).strip(" :[]")
                if label and INSTRUCTOR_LINE.match(f"{label} x") and len(label.split()) <= 6 and INSTRUCTOR_LABEL.search(label):
                    names: list[str] = []
                    for line in (line.strip(" :") for value in cells[index + 1:] for line in value.split("\n")):
                        if not line:
                            continue
                        if names and len(line.split()) == 1 and "(" not in line and not names[-1].endswith(")"):
                            names[-1] = f"{names[-1]} {line}"  # a wrapped surname ("BHANU VARDHAN REDDY" / "KUNCHARAM")
                        else:
                            names.append(line)
                    if names:
                        groups.append([bool(re.search(r"charge", label, re.I)), ", ".join(names)])
                        page_found = page_found or page
                    break
    return groups, page_found


def extract_instructors(block: list[Line], tables: list[PageTable] | None = None) -> tuple[list[str], int | None]:
    """Instructor names from labelled header lines, instructor-in-charge first, as printed.

    why: lists often wrap ("Lab Instructors: A, B,\\n C, D"), so unlabelled lines after an
    instructor label are part of it until the next 'Label :' line.
    """
    groups: list[list] = []  # [is_ic_label, text] per instructor label, continuation lines joined
    page_found: int | None = None
    current: list | None = None
    for page, line in block:
        line = re.sub(r"^\s*\[([^\]]+)\]", r"\1", line)  # "[Team of Instructors] : ..." -> "Team of Instructors : ..."
        labelled = LABEL_LINE.match(line)
        if not (labelled and INSTRUCTOR_LABEL.search(labelled[1])):
            labelled = INSTRUCTOR_LINE.match(line) or labelled
        if labelled:
            label, value = labelled[1], labelled[2]
            if not INSTRUCTOR_LABEL.search(label) or re.search(r"course\s*(?:no|number|title|name)|email|website|room|chamber", label, re.I):
                current = None
                continue
            current = [bool(re.search(r"charge", label, re.I)), value]
            groups.append(current)
            page_found = page_found or page
        elif current is not None and not SENTENCE_WORDS.search(re.sub(r"\(.*?\)", " ", line)):
            # why: a wrapped list may break inside a name ("Sandhya A" / "Marathe"), so join before splitting.
            # ": Prof. B (email)" lines, or a line after a closed "(...)", start a new name.
            # A wrapped surname is one word ("Sandhya A" / "Marathe"); two or more words are the next name.
            separator = ", " if line.lstrip().startswith(":") or line.lstrip()[:1].isdigit() or current[1].rstrip().endswith(")") \
                or len(line.split(",")[0].split()) >= 2 and len(current[1].split(",")[-1].split()) >= 2 else " "
            current[1] = f"{current[1]}{separator}{line.lstrip(' :')}"
        else:
            current = None  # a sentence ends the instructor list (e.g. the course description follows it)
    from_text = ordered_names(groups)
    table_groups_found, table_page = instructor_groups_from_tables(tables or [])
    from_tables = ordered_names(table_groups_found)
    # why: header tables list one name per line, which flattened text merges ("Rejaul Raja Piyush"); but some
    # tables hold the names in another row, so the table reading is used only when it finds more names.
    if len(from_tables) > len(from_text):
        return from_tables, table_page
    return from_text, page_found


def ordered_names(groups: list[list]) -> list[str]:
    """[is_ic_label, text] groups -> unique names, instructor-in-charge first."""
    in_charge: list[str] = []
    others: list[str] = []
    for is_ic_label, text in groups:
        for name, marked_ic in split_names(text):
            (in_charge if is_ic_label or marked_ic else others).append(name)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in in_charge + others:
        key = HONORIFIC.sub("", name).lower().replace(" ", "")
        if key not in seen:
            seen.add(key)
            ordered.append(name)
    return ordered


# ---------------------------------------------------------------- about

def join_wrapped(lines: list[str]) -> list[tuple[str, bool]]:
    """Group lines into (text, is_list_item). A marker line starts a new item; other lines continue the last."""
    items: list[list] = []
    for line in lines:
        marked = bool(LIST_MARKER.match(line)) and not re.match(r"^\s*[A-Z][a-z]", line)
        if marked or not items:
            items.append([line, marked])
        else:
            previous = items[-1][0]
            items[-1][0] = previous[:-1] + line if re.search(r"[a-z]-$", previous) and line[:1].islower() else f"{previous} {line}"
    return [(text, marked) for text, marked in items]


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text) if part.strip()]


def build_about(sections: dict[str, list[Line]], models, removed_log: Counter) -> tuple[str | None, int | None]:
    """Description + scope/objectives + learning outcomes as one text, lead-in lines and markers removed.

    A candidate lead-in is an unmarked sentence that doesn't end with a full stop. It is dropped if
    - structure: it ends with ':' or a dash and a marked list follows ("...students will be able to:"), or
    - backup: it has no closing punctuation and is very close to a header example (embedding similarity).
    why: a sentence ending with '.' is always kept; real content like "The aim of this course is to carry
    out a research project." looks header-like to the embedding, and "Out of various techniques, emphasis
    will be on GA, ANN and Fuzzy Systems" precedes a list without being a header.
    Every kept word comes from the handout.
    """
    pieces: list[str] = []
    page: int | None = None
    for key in ("description", "objectives", "learning_outcomes"):
        lines = sections.get(key, [])
        if not lines:
            continue
        page = page or lines[0][0]
        items = join_wrapped([line for _, line in lines])
        units: list[tuple[str, bool]] = []  # (sentence, is_list_item)
        for index, (text, marked) in enumerate(items):
            if marked:
                units.append((LIST_MARKER.sub("", text, count=1).strip(), True))
                continue
            sentences = split_sentences(text)
            next_is_list = index + 1 < len(items) and items[index + 1][1]
            if next_is_list and sentences and len(sentences[-1].split()) <= 30 and re.search(r"[:\-–]\s*\d?$", sentences[-1]):
                removed_log[sentences[-1]] += 1
                sentences = sentences[:-1]
            units += [(sentence, False) for sentence in sentences]

        candidates = [(i, text) for i, (text, marked) in enumerate(units)
                      if not marked and len(text.split()) <= HEADER_MAX_WORDS and not text.rstrip().endswith((".", "!", "?"))]
        similarity = models.max_similarity([text for _, text in candidates], HEADER_EXAMPLES) if candidates else []
        drop = {i for (i, _), score in zip(candidates, similarity) if score >= HEADER_SIMILARITY}
        for i in drop:
            removed_log[units[i][0]] += 1
        for i, (text, _) in enumerate(units):
            text = text.strip()
            if i in drop or not text:
                continue
            pieces.append(text if text.endswith((".", "!", "?")) else text.rstrip(":;,") + ".")
    return (" ".join(pieces) or None), page


# ---------------------------------------------------------------- tables

def column_index(names: list[str], pattern: str, exclude: tuple[int | None, ...] = ()) -> int | None:
    """First column whose header matches; also tried without spaces (letter-spaced "M odule")."""
    return next((i for i, name in enumerate(names)
                 if i not in exclude and (re.search(pattern, name) or re.search(pattern, name.replace(" ", "")))), None)


def cell_text(cell: str | None) -> str:
    """Cell with wrapped lines joined ('Fluid Me-\\nchanics' -> 'Fluid Mechanics')."""
    return clean(re.sub(r"(\w)-\n(?=[a-z])", r"\1", cell or ""))


def row_names(row: Row) -> list[str]:
    return [cell_text(cell).lower() for cell in row]


def is_real_header(row: Row, is_header) -> bool:
    """A header row: passes is_header, and its cells are short labels that don't start with a number."""
    cells = [cell_text(cell) for cell in row if cell_text(cell)]
    return is_header(row_names(row)) and len(cells) >= 2 and all(len(cell) <= 40 and not cell[:1].isdigit() for cell in cells)


def split_header(table: list[Row], is_header) -> tuple[Row | None, list[Row]]:
    """(header row, body rows), or (None, table) if the table doesn't start with such a header.

    The header may span up to HEADER_MAX_ROWS rows, merged cell by cell. why: many plans print a
    multi-row header ("Module" / "Number", "Text/Ref" / "Book" / "Chapter #"), sometimes under a blank
    first row. Label rows are merged until the header passes is_header; after that only label fragments
    (mostly empty, short, no digits: "Number", "Chapter #") are merged, so the first body row is kept.
    """
    def label_row(cells: list[str]) -> bool:
        # why: labels have few numbers and no ranges ("Marks (300)", "Reference (21st Ed., 2017)");
        # "Module- 7 28-38 Strategies" and "Lectures 20-24: ..." are body rows.
        # A leading section number is fine ("5. Evaluation Scheme" printed inside the first cell).
        cells = [re.sub(r"^\d{1,2}\.\s*(?=[A-Za-z])", "", cell) for cell in cells if cell]
        return all(len(cell) <= 80 and not cell[:1].isdigit() and len(re.findall(r"\d+", cell)) <= 2
                   and not re.search(r"\d\s*[-–]\s*\d", cell) for cell in cells)

    def fragment(cells: list[str]) -> bool:
        return sum(not cell for cell in cells) * 2 >= len(cells) and all(len(cell) <= 12 and not re.search(r"\d", cell) for cell in cells)

    merged = [""] * len(table[0])
    for index, row in enumerate(table[:HEADER_MAX_ROWS + 1]):
        cells = [cell_text(cell) for cell in row] + [""] * (len(merged) - len(row))
        found = sum(bool(cell) for cell in merged) >= 2 and is_header([name.lower() for name in merged])
        if found and not fragment(cells):
            return merged, table[index:]
        if not found and (not label_row(cells) or index == HEADER_MAX_ROWS):
            return None, table
        merged = [clean(f"{old} {new}") for old, new in zip(merged, cells)]
    found = sum(bool(cell) for cell in merged) >= 2 and is_header([name.lower() for name in merged])
    return (merged, table[HEADER_MAX_ROWS + 1:]) if found else (None, table)


def strong_header(header: Row) -> bool:
    """At least two cells are plain header labels: short, no digits, a header word ("Module Number", "Reference")."""
    labels = [cell for cell in header if cell and len(cell.split()) <= 5 and not re.search(r"\d", cell)
              and re.search(HEADER_WORDS, cell, re.I)]
    return len(labels) >= 2


def table_groups(tables: list[PageTable], is_header) -> list[tuple[list[Row], int]]:
    """Every table whose header passes is_header, merged with its continuations on later pages.

    Each group is [header row] + body rows. why (the truncated-topics bug): a table continued on the next
    page often repeats its header, or starts with a body row that happens to contain a header word
    ("lecture"). Continuations are tables of about the same width (pdfplumber sometimes adds an empty
    column) that repeat the header or start without any known header (an evaluation table right after
    the course plan is not its continuation).
    """
    groups: list[tuple[list[Row], int]] = []
    current: list[Row] | None = None
    for page, table in tables:
        header, body = split_header(table, is_header)
        if header is not None and current is not None and not strong_header(header):
            header, body = None, table  # a body row with a header word ("TB Chapter 4"): a continuation
        if current is not None and abs(len(table[0]) - len(current[0])) <= 1:
            after_total = bool(re.match(r"total", " ".join(cell_text(cell) for cell in current[-1]).strip(), re.I))
            if header is not None and [name.lower() for name in header] == row_names(current[0]) and not after_total:
                current.extend(body)
                continue
            contact = re.search(r"e-?mail|phone|chamber", " ".join(row_names(table[0])))  # instructor contact table
            if header is None and not contact and not any((found := split_header(table, other)[0]) and strong_header(found)
                                          for other in (is_plan_header, is_evaluation_header)):
                current.extend(table)
                continue
        if header is not None:
            current = [header, *body]
            groups.append((current, page))
        else:
            current = None
    return groups


def neighbour_cell(names: list[str], row: Row, index: int | None) -> str:
    """The row's value under a header; falls back to a blank-headed neighbour (merged header cells)."""
    if index is None:
        return ""
    for candidate in (index, index - 1, index + 1):
        # why: continuation tables can be one column wider/narrower than the header row.
        if 0 <= candidate < len(row) and (candidate == index or (candidate < len(names) and not names[candidate])):
            value = cell_text(row[candidate])
            if value:
                return value
    return ""


def is_plan_header(names: list[str]) -> bool:
    return any(PLAN_HEADER.search(name) for name in names) and not any(EVALUATION_HEADER.search(name) for name in names)


def is_evaluation_header(names: list[str]) -> bool:
    return column_index(names, EVALUATION_HEADER.pattern) is not None and column_index(names, r"component|evaluation|duration|date") is not None


def split_topic_cell(text: str) -> list[str]:
    """One cell -> topics: split on bullets and on inner 'L3-L6:' style lecture prefixes, prefixes removed."""
    parts = []
    for chunk in BULLETS.split(text):
        parts += re.split(INNER_LECTURE, chunk)
    topics = []
    for part in parts:
        part = clean(TOPIC_PREFIX.sub("", clean(part))).strip(" ,;:-–")
        # why: "(total lectures 5)" style cells are counts, not topics.
        if part and re.search(r"[A-Za-z]{3}", part) and not re.fullmatch(r"(class )?slides?|tutorials?|revision|-|\(.*\)|(s|sl|sr|expt?|exp)\.? ?no\.?|total|learning outcomes?", part, re.I):
            topics.append(part)
    return topics


def extract_topics(tables: list[PageTable]) -> tuple[list[str], int | None, bool]:
    """(topics, page, found_a_plan_table): module titles + lecture topics from every course-plan table."""
    topics: list[str] = []
    seen: set[str] = set()
    first_page: int | None = None
    fallback: list[str] = []  # from objectives/outcomes columns, used only if no table gives real topics
    for rows, page in table_groups(tables, is_plan_header):
        names = row_names(rows[0])
        module_col = column_index(names, MODULE_COLUMN)
        numbers = tuple(i for i, name in enumerate(names) if re.search(NUMBER_COLUMN, name))
        topic_col = next((column_index(names, pattern, (module_col, *numbers)) for pattern in TOPIC_COLUMNS
                          if column_index(names, pattern, (module_col, *numbers)) is not None), None)
        if module_col is None and topic_col is None:
            topic_col = column_index(names, OBJECTIVE_COLUMN, numbers)
        if module_col is None and topic_col is None:
            continue
        first_page = first_page or page
        found = [topic for row in rows[1:] for column in (module_col, topic_col)
                 for topic in split_topic_cell(neighbour_cell(names, row, column))]
        if not found:  # the module/lecture columns hold only numbers ("1", "1-2"): the objectives/outcomes may do
            outcome_col = column_index(names, OBJECTIVE_COLUMN, (module_col, topic_col, *numbers))
            fallback += [topic for row in rows[1:] for topic in split_topic_cell(neighbour_cell(names, row, outcome_col))]
        for topic in found:
            if topic.lower() not in seen:
                seen.add(topic.lower())
                topics.append(topic)
    return topics or list(dict.fromkeys(fallback)), first_page, first_page is not None


# A text-plan row: topic, then a lecture range and a book reference ("Ethics in stem cell research 12-14 TB:3").
TEXT_PLAN_ROW = re.compile(r"^(?P<topic>[A-Z(].*?)\s+\d{1,2}\s*[-–]\s*\d{1,2}\s+(?:TB|RB|T\s*\d|R\s*\d|Ch)")
# A module line in an unruled plan ("(iv) Musical Scales L-4.1- ...", "Module 3: Heat transfer").
TEXT_PLAN_MODULE = re.compile(r"^(?:\(([ivx]{1,5})\)|module\s*\d+\s*[:.\-–]?)\s*(?P<topic>[A-Za-z].*?)(?=\s+(?:L-?\s*)?\d+\.\d+|\s*$)", re.I)


def topics_from_text(lines: list[Line]) -> tuple[list[str], int | None, bool]:
    """Fallback for course plans printed without table lines: rows ending in a lecture range + book
    reference (wrapped lower-case lines continue the row), else numbered module lines. Kept as printed.
    The third value is False for module lines: in an unruled multi-column layout their wrapped words can't
    be told apart from the next column, so those topics may be cut ("Philosophy and")."""
    topics: list[str] = []
    page: int | None = None
    open_row = False  # only lines right after a row continue it (not a footnote further down)
    for line_page, line in lines:
        row = TEXT_PLAN_ROW.match(line)
        if row:
            topics.append(clean(row["topic"]))
            page = page or line_page
            open_row = True
        elif open_row and line[:1].islower() and not re.search(r"\d{1,2}\s*[-–]\s*\d{1,2}", line):
            topics[-1] = f"{topics[-1]} {line}"
        else:
            open_row = False
    if len(topics) >= TOPICS_MIN_EXPECTED:
        return list(dict.fromkeys(topics)), page, True
    topics = []
    for line_page, line in lines:
        module = TEXT_PLAN_MODULE.match(line)
        if module and len(module["topic"].split()) <= 12:
            topics.append(clean(module["topic"]).strip(" ,;:-–"))
            page = page or line_page
    return list(dict.fromkeys(topic for topic in topics if topic)), page, False


def topics_from_label_tables(tables: list[PageTable]) -> tuple[list[str], int | None]:
    """Plans printed as one small key-value table per session ("Session 1-2" / "Themes: ...")."""
    topics: list[str] = []
    page: int | None = None
    for table_page, table in tables:
        for row in table:
            cells = [cell_text(cell) for cell in row]
            if len(cells) >= 2 and re.fullmatch(r"themes?|topics?( covered)?", cells[0], re.I) and cells[1]:
                topics.append(cells[1])
                page = page or table_page
    return topics, page


def parse_weight(text: str, marks_then_percent: bool = False) -> float | None:
    """Weight cell -> number, as printed. Handles every form seen in the handouts:

    '25%', '25', '50 (25 %)' -> 25 (a percent beats marks); '1 5 %' -> 15 (letter-spaced);
    '10% + 5%', '20 + 10', '10 10', '20 30 20' -> the sum (a component split into parts);
    '10x2=20 Marks' -> 20; '60 /30' under a "Marks/Weightage (%)" header -> 30.
    """
    text = re.sub(r"(\d)\s+(\d)(?=\s*%)", r"\1\2", text)
    percents = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
    if len(percents) > 1 and re.search(r"%\s*\+", text):
        return sum(float(value) for value in percents)
    if percents:
        return float(percents[0])
    if marks_then_percent and (pair := re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)):
        return float(pair[2])
    if equals := re.search(r"=\s*(\d+(?:\.\d+)?)", text):
        return float(equals[1])
    if re.fullmatch(r"\s*\d+(?:\.\d+)?(?:\s*\+?\s*\d+(?:\.\d+)?)+\s*[#*]*\s*", text):
        return sum(float(value) for value in re.findall(r"\d+(?:\.\d+)?", text))
    number = re.search(r"\d+(?:\.\d+)?", text)
    return float(number[0]) if number else None


def duration_minutes(text: str) -> int | None:
    """'90 min' -> 90, '1.5 hours' -> 90, '3 hrs' -> 180; '-', take-home, TBA or unclear -> None."""
    text = text.lower()
    if not text or re.search(r"take[\s-]?home", text) or re.fullmatch(r"\s*(tba|tbd|n/?a|[-–]+)\s*", text):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(min|mins|minutes|mints|hr|hrs|hour|hours|h)\b", text)
    if match:
        return round(float(match[1]) * (60 if match[2].startswith("h") else 1))
    bare = re.fullmatch(r"\s*(\d+)\s*\.?\s*", text)
    return int(bare[1]) if bare and int(bare[1]) >= 30 else None  # why: a bare "90" is minutes; a bare "3" is ambiguous


def component_kind(name: str) -> str:
    lowered = name.lower()
    return next((kind for kind, pattern in COMPONENT_KINDS.items() if re.search(pattern, lowered)), "other")


def book_nature(text: str) -> str | None:
    """'Closed Book' / 'CB' -> 'CB'; 'Open Book' / 'OB' -> 'OB'; both or neither -> None."""
    closed = re.search(r"closed?[\s-]*book|\bCB\b|closed", text, re.I)
    opened = re.search(r"open[\s-]*book|\bOB\b|open", text, re.I)
    return None if bool(closed) == bool(opened) else "CB" if closed else "OB"


def group_flag(text: str) -> bool | None:
    group, individual = re.search(r"\bgroups?\b|\bteams?\b", text, re.I), re.search(r"individual", text, re.I)
    return None if bool(group) == bool(individual) else bool(group)


def clean_remarks(text: str) -> str | None:
    return clean(BULLETS.sub(" ", text)).strip(" ;,") or None


def make_component(name: str, weight: float | None, duration: str, date: str, remarks: str, nature_text: str = "") -> Component:
    name = re.sub(r"^\[\s*|\s*\]$", "", name).replace("] ", " ").replace(" [", " ")  # template brackets "[Mid Semester]"
    return Component(
        name=name,
        kind=component_kind(name),
        weightage_percent=weight,
        duration_minutes=duration_minutes(duration),
        nature=book_nature(f"{name} {nature_text} {remarks}"),
        is_group=group_flag(f"{name} {remarks}"),
        date=date or None,
        remarks=clean_remarks(remarks),
    )


def parse_evaluation_table(rows: list[Row]) -> tuple[list[Component], bool, float | None]:
    """(components, weights_are_marks, printed_total). Columns are found by header name."""
    names = row_names(rows[0])
    serial = tuple(i for i, name in enumerate(names) if re.search(NUMBER_COLUMN, name) or name == "ec")
    weight_col = column_index(names, r"weigh|%|wt|percent")
    if weight_col is None:
        weight_col = column_index(names, r"marks")
    weight_name = names[weight_col] if weight_col is not None else ""
    marks_then_percent = "mark" in weight_name and "weigh" in weight_name
    weight_cells = [neighbour_cell(names, row, weight_col) for row in rows[1:]]
    # why: "Weightage (GT=200]" columns hold "10 Marks": marks even though the header says weightage.
    marked_cells = sum(bool(re.search(r"marks?\b", cell, re.I)) and "%" not in cell for cell in weight_cells)
    is_marks = weight_col is not None and "%" not in weight_name and (
        "mark" in weight_name and "weigh" not in weight_name or marked_cells * 2 > len([cell for cell in weight_cells if cell]))
    name_col = column_index(names, r"component|evaluation|name|test|type", (*serial, weight_col))
    details_col = column_index(names, r"^details?$", (*serial, weight_col))
    if details_col is not None:  # "Component: Continuous Evaluation (30 %)" groups rows named in "Details"
        name_col = details_col
    if name_col is None:
        name_col = next((i for i in range(len(names)) if i not in (*serial, weight_col)), None)
    if name_col is None:
        return [], False, None
    duration_col = column_index(names, r"duration|time\b", (name_col, weight_col))
    date_col = column_index(names, r"date", (name_col, weight_col, duration_col))
    nature_col = column_index(names, r"nature|mode|open|close", (name_col, weight_col))
    remarks_col = column_index(names, r"remark|comment", (name_col, weight_col, nature_col))

    components: list[Component] = []
    printed_total: float | None = None
    for row in rows[1:]:
        name = neighbour_cell(names, row, name_col)
        if not re.search(r"[A-Za-z]{3}", name):  # "(%)", "5%": a misaligned or header-fragment row
            continue
        weight = parse_weight(neighbour_cell(names, row, weight_col), marks_then_percent)
        if re.match(r"(course |grand )?total\b", name, re.I):
            printed_total = weight
            continue
        duration, date = neighbour_cell(names, row, duration_col), neighbour_cell(names, row, date_col)
        remarks, nature_text = neighbour_cell(names, row, remarks_col), neighbour_cell(names, row, nature_col)
        if weight is None and components and not (duration or date):
            # why: a row with only text is a wrapped line or footnote of the row above, not a component.
            previous = components[-1]
            if name[0].isalpha() and ":" not in name and len(name.split()) <= 4:  # the name wrapped onto this row
                previous.name = f"{previous.name} {name}"
                previous.kind = component_kind(previous.name)
            elif not name[0].isalpha():  # "*Best 2 of 3", "(#...)": a footnote about the row above
                previous.remarks = clean_remarks(f"{previous.remarks or ''} {name}")
            continue  # other text rows ("Criterion for NC: ...") are not about this component
        components.append(make_component(name, weight, duration, date, remarks, nature_text))
    return components, is_marks, printed_total


def text_weight(rest: str) -> float | None:
    """The weight among the numbers after a component name: a percent if there is one, else the first number
    that isn't a duration ("1.5 h", "90 mints"), a date ("09/10") or an ordinal ("2nd week")."""
    percent = re.search(r"(\d+(?:\.\d+)?)\s*%", rest)
    if percent:
        return float(percent[1])
    if parts := re.match(r"\s*(\d+(?:\.\d+)?(?:\s*\+\s*\d+(?:\.\d+)?)+)(?!\s*(?:h|hrs?|min))", rest):  # "10+10"
        return sum(float(value) for value in re.findall(r"\d+(?:\.\d+)?", parts[1]))
    for number in re.finditer(r"(?<![\d/.])(\d+(?:\.\d+)?)(?![\d/]|\.\d)", rest):
        after = rest[number.end():]
        if not re.match(r"\s*(?:h|hrs?|hours?|min|mins|minutes|mints|st|nd|rd|th)\b", after, re.I):
            return float(number[1])
    return None


def parse_evaluation_text(lines: list[str]) -> list[Component]:
    """Fallback for evaluation 'tables' without ruling lines: one component per line with a weight.

    Handles 'Mid Semester Exam 20 90 min', 'Quiz 15% TBA', '● 10% Class Participation', '[Mid Semester] [30%]',
    'Lab Related Activities – 1 25 6th week', 'Mid-semester evaluation 1.5 h 25 09/10 AN2'.
    Only used if the weights add up to 100 (or, flagged, if there are several rows).
    """
    components: list[Component] = []
    percent_rows: list[bool] = []
    for line in lines:
        line = line.replace("[", "").replace("]", "")
        text = re.sub(r"^[\W\d]{0,4}?(?=[A-Za-z%\d])", "", line) if not re.match(r"^\W*\d+\s*%", line) else line
        percent_first = re.match(r"^\W*(\d+(?:\.\d+)?)\s*%\s*([A-Za-z][^:]*)", text)
        # name = text up to the first number; a trailing "– 1" / "- 2" is part of the name when a number follows
        # "Quiz – 2 15%", "Tutorial 1 5%": a single digit right after the name is part of it when a percent follows
        name_first = re.match(r"^([A-Za-z][^\d%]*?[-–]\s*\d{1,2})\s+(?=\d)(.*)$", text) \
            or re.match(r"^((?:[A-Za-z][^\d%]*?\s)?(?:quiz|tutorial|test|assignment|seminar|lab|viva|project|report|evaluative|exam)(?:e?s)?"
                        r"\s*#?\s*\d)\s+(?=\d+(?:\.\d+)?\s*%)(.*)$", text, re.I) \
            or re.match(r"^([A-Za-z][^\d%]*?)[\s:\-–….]*(?=\d)(.*)$", text)
        rest = ""
        if percent_first:
            weight, name = float(percent_first[1]), percent_first[2]
        elif name_first:
            name, rest = name_first[1], name_first[2]
            weight = text_weight(rest)
            if weight is None:
                continue
        else:
            continue
        name = re.sub(r"^\(?([a-h]|[ivx]{1,4})\)\s*", "", clean(name)).strip(" :-–")
        # why: lowercase starts are wrapped sentences; long names are sentences ("NC ... below 30% of ...").
        if not name or name[0].islower() or re.match(r"(total|component|evaluation|note)\b", name, re.I) \
                or weight > 100 or len(name.split()) > 8:
            continue
        components.append(make_component(name, weight, rest, "", rest))
        percent_rows.append("%" in line)
    if sum(percent_rows) * 2 > len(percent_rows):
        components = [component for component, has_percent in zip(components, percent_rows) if has_percent]
    return components


def weight_total(components: list[Component]) -> float | None:
    """Sum of the stated weights; None if there are none. Components without a weight are ignored."""
    weights = [component.weightage_percent for component in components if component.weightage_percent is not None]
    return round(sum(weights), 2) if weights else None


NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


def best_of_adjustment(components: list[Component], context: str, total: float) -> tuple[float, str] | None:
    """(weight not counted, note) for "best N out of M" schemes, e.g. three 15% quizzes where the best two count.

    why: such tables list all M quizzes, so the printed weights add up to more than 100.
    """
    # "best two (quizzes) out of three", or "2 Best Scores out of the 4" (tried separately: both can match one phrase)
    patterns = (r"best\s+(?:of\s+)?(\w+)\s+(?:\w+\s+){0,3}?(?:out\s+)?of\s+(?:the\s+)?(\w+)",
                r"(\w+)\s+best\s+(?:\w+\s+){0,2}?(?:out\s+)?of\s+(?:the\s+)?(\w+)")
    for match in (match for pattern in patterns for match in re.finditer(pattern, context, re.I)):
        words = list(match.groups())
        counted, listed = (NUMBER_WORDS.get(word.lower()) or (int(word) if word.isdigit() else None) for word in words)
        if not counted or not listed or counted >= listed:
            continue
        for kind in {component.kind for component in components}:
            weights = [c.weightage_percent for c in components if c.kind == kind and c.weightage_percent is not None]
            # only an adjustment that makes the scheme add up to 100 is taken
            if len(weights) == listed and len(set(weights)) == 1 and abs(total - (listed - counted) * weights[0] - 100) <= WEIGHT_TOLERANCE:
                return (listed - counted) * weights[0], f"best_{counted}_of_{listed}_{kind}"
    return None


def evaluation_candidates(doc: Document, sections: dict[str, list[Line]]):
    """Possible evaluation schemes as (source, components, is_marks, printed_total, page), best source first:
    each evaluation table; all tables together (a scheme split over two differently-shaped tables, never for
    marks tables, which always normalise to 100); then the section text with and without table masking."""
    groups = table_groups(doc.tables, is_evaluation_header)
    parsed = [(*parse_evaluation_table(rows), page) for rows, page in groups]
    candidates = [(f"table{index}", *result) for index, result in enumerate(parsed)]
    if len(parsed) > 1 and not any(result[1] for result in parsed):
        candidates.append(("tables", copy.deepcopy([c for result in parsed for c in result[0]]), False, None, parsed[0][3]))
    for section in (sections.get("evaluation", []), split_sections(doc.header_lines).get("evaluation", [])):
        if section:
            candidates.append(("text", parse_evaluation_text([line for _, line in section]), False, None, section[0][0]))
    return candidates


def drop_subtotal_rows(components: list[Component]) -> str | None:
    """Remove a "Quiz Best 2 of 3 (10%)" row that restates the quizzes listed above it; returns a note or None."""
    for row in components:
        match = re.search(r"best\s+(\w+)\s+(?:out\s+)?of\s+(\w+)", row.name, re.I)
        if not match or row.weightage_percent is None:
            continue
        counted, listed = (NUMBER_WORDS.get(word.lower()) or (int(word) if word.isdigit() else 0) for word in match.groups())
        parts = [c for c in components if c is not row and c.kind == row.kind and c.weightage_percent is not None]
        if counted and len(parts) == listed and len({c.weightage_percent for c in parts}) == 1 \
                and abs(parts[0].weightage_percent * counted - row.weightage_percent) < 0.01:
            components.remove(row)
            return f"best_{counted}_of_{listed}_{row.kind}"
    return None


def settle_weights(components: list[Component], is_marks: bool, printed_total: float | None, context: str):
    """(total, issues, notes) after marks -> percent conversion and the best-of adjustment."""
    issues: list[str] = []
    notes: list[str] = []
    drop_subtotal_rows(components)  # then the best-of rule below counts the quizzes it summarised

    total = weight_total(components)
    if is_marks and total:
        scale = printed_total or total
        for component in components:
            if component.weightage_percent is not None:
                component.weightage_percent = round(component.weightage_percent * 100 / scale, 2)
        notes.append("weightage_in_marks_converted")
        total = weight_total(components)
    if total is not None and abs(total - 100) > WEIGHT_TOLERANCE and (best := best_of_adjustment(components, context, total)):
        notes.append(best[1])
        total = round(total - best[0], 2)
    if total is None:
        issues.append("weightage_missing")
    elif abs(total - 100) > WEIGHT_TOLERANCE:
        issues.append(f"weightage_sum_{total:g}")
    elif any(component.weightage_percent is None for component in components):
        notes.append("components_without_weight")  # e.g. "Assignments (non-evaluative)": the rest add up to 100
    return total, issues, notes


def extract_evaluation(doc: Document, sections: dict[str, list[Line]]) -> tuple[list[Component], int | None, list[str], list[str]]:
    """(components, page, issues, notes).

    Among candidates whose weights add up to 100, tables beat text, then the one with most weighted components
    wins (the thesis handouts print a 3-row mid-semester form before the full 6-row scheme). If none adds up,
    the first table (or a text result of MIN_TEXT_COMPONENTS+ rows) is kept with its issue; shorter text
    results are dropped (they are usually NC-rule sentences like "Criterion for NC: 30% of the average").
    If the first table doesn't add up but a later one does, the scheme may span both (two-module courses):
    kept, but flagged.
    """
    # why: best-of notes are often printed inside the table ("Note: 2 Best Scores out of the 4 Quizzes")
    context = " ".join(line for _, line in split_sections(doc.header_lines).get("evaluation", []))
    passing = []
    first = None
    for order, (source, components, is_marks, printed_total, page) in enumerate(evaluation_candidates(doc, sections)):
        if not components:
            continue
        remarks = " ".join(f"{c.name} {c.remarks or ''}" for c in components)
        total, issues, notes = settle_weights(components, is_marks, printed_total, f"{remarks} {context}")
        weighted = sum(c.weightage_percent is not None for c in components)
        if not issues:
            passing.append(((source != "text", weighted, -order), source, components, page, notes))
        elif first is None and (source.startswith("table") or len(components) >= MIN_TEXT_COMPONENTS):
            first = (source, components, page, issues, notes)
    if passing:
        _, source, components, page, notes = max(passing, key=lambda item: item[0])
        issues = ["evaluation_partial"] if first and first[0] == "table0" and source.startswith("table") and source not in ("table0", "tables") else []
        return components, page, issues, notes
    if first:
        return first[1], first[2], first[3], first[4]
    return [], None, ["evaluation_not_found"], []


# ---------------------------------------------------------------- model-based fields

def to_sentences(lines: list[Line]):
    """Section lines -> Sentence objects with the page each sentence starts on.

    why: header blocks have no full stops ("Course No: ...", "Date: ..."), so joining every line
    makes giant fake sentences. A line joins the previous one only if it clearly continues it.
    """
    from handout_models import Sentence
    units: list[list] = []  # [text, page]
    previous_line = ""
    for page, line in lines:
        if not line:  # a removed heading / header line: nothing continues across it
            units.append(["", page])
            previous_line = ""
            continue
        previous = units[-1][0] if units else ""
        # why: a long line without end punctuation was wrapped mid-sentence ("... Test/Examination. No" /
        # "Make-up for the Class Tests"); short ones are header-style lines ("Course No: ...").
        continues = bool(units) and previous and not LIST_MARKER.match(line) and not re.search(r"[.:;!?][”\"’']?$", previous) and (
            line[:1].islower() or len(previous_line) >= WRAPPED_LINE_CHARS
            or previous.endswith((",", "-", "&", " and", " or", " the", " of", " to", " for")))
        previous_line = line
        if continues:
            units[-1][0] = previous[:-1] + line if previous.endswith("-") and line[:1].islower() else f"{previous} {line}"
        else:
            units.append([line, page])
    sentences = []
    for text, page in units:
        for part in re.split(r"(?:(?<=[.!?])|(?<=[.!?][”\"’']))\s+(?=[A-Z(\d“\"])", BULLETS.sub(" \n ", text)):
            for piece in part.split(" \n "):
                if len(piece.split()) >= 3:
                    sentences.append(Sentence(clean(piece), page))
    return sentences


def body_lines(doc: Document) -> list[Line]:
    """The whole handout minus its header block and heading lines: the fallback pool when a field has no section.

    why: header lines ("Course Handout Part II", "In addition to Part I ...", instructor lists) are never
    policy sentences but score high on embedding similarity.
    """
    header = set(header_block(doc.lines))
    # why: dropped lines become "" so that to_sentences doesn't join the text on either side of them.
    return [(page, line if (page, line) not in header and match_heading(line) is None else "") for page, line in doc.lines]


def policy_sentences(section: list[Line] | None, doc: Document, field_name: str):
    """Sentences of the field's section; without one, body sentences containing the field's word stem."""
    if section:
        return to_sentences(section)
    return [sentence for sentence in to_sentences(body_lines(doc)) if re.search(FALLBACK_KEYWORDS[field_name], sentence.text, re.I)]


def method_entry(decision, method: str = "nli") -> dict:
    return {"method": method, "confidence": round(decision.confidence, 3) if decision.confidence else None,
            "evidence": decision.evidence.text if decision.evidence else None}


def settle(decision, field_name: str, stats: Counter) -> tuple[bool | None, dict, list[str]]:
    """(value, method entry, issues) for one NLI decision.

    confident entail/contradict -> True/False; otherwise a confident "neutral" means the text mentions the
    topic but doesn't say (null, no issue, as the spec's neutral -> null); anything else is uncertain.
    """
    from handout_models import Decision
    if decision.confidence and decision.confidence >= NLI_MIN_CONFIDENCE:
        stats[f"{field_name}:decided"] += 1
        return decision.value, method_entry(decision), []
    if decision.neutral_confidence >= NLI_MIN_CONFIDENCE:
        stats[f"{field_name}:neutral"] += 1
        return None, {"method": "nli", "confidence": round(decision.neutral_confidence, 3), "evidence": None,
                      "label": "neutral"}, []
    stats[f"{field_name}:uncertain"] += 1
    return None, method_entry(Decision(None, decision.confidence, decision.evidence)), [f"{field_name.replace('.', '_')}_uncertain"]


def names_a_component(text: str) -> bool:
    # why: "no makeup for continuous evaluation" is about quizzes/labs etc., not a rule for every component.
    return any(re.search(pattern, text.lower()) for pattern in [*COMPONENT_KINDS.values(), r"continuous evaluation"])


def extract_attendance(models, sentences, section_text: str | None, stats: Counter) -> tuple[dict, dict, int | None, list[str]]:
    """(attendance, method entry, page, issues). Locate -> not mentioned / decide -> value, neutral or uncertain.

    section_text is the Attendance section when the handout has one; it is the attendance text even
    when it is too vague to decide ("As per guidelines by AUGS/AGSR Division.").
    """
    from handout_models import Decision
    located = models.locate(sentences, FIELD_DESCRIPTIONS["attendance"], TOP_K_SENTENCES)
    relevant = [item for item in located if item.similarity >= MIN_SIMILARITY["attendance"]]
    if not relevant:
        stats["attendance.required:not_mentioned"] += 1
        return {"required": None, "percent": None, "text": section_text}, method_entry(Decision(None, None, None)), None, []
    ordered = sorted(relevant, key=lambda item: sentences.index(item.sentence))
    text = section_text or " ".join(item.sentence.text for item in ordered)
    percent = re.search(r"(?:minimum|at\s*least|atleast|min\.?)\s*(?:of\s*)?(\d{2,3})\s*%|(\d{2,3})\s*%\s*(?:attendance|of\s*(?:the\s*)?(?:classes|lectures|sessions))", text, re.I)
    decision = models.decide([item.sentence for item in relevant], HYPOTHESES["attendance.required"])
    value, entry, issues = settle(decision, "attendance.required", stats)
    attendance = {"required": value, "percent": int(percent[1] or percent[2]) if percent else None, "text": text}
    page = (decision.evidence or ordered[0].sentence).page
    return attendance, entry, page, issues


def makeup_clauses(sentences, from_section: bool):
    """Policy sentences -> single clauses, each a premise of its own.

    why: "No make-up for quizzes; however, for other components make-up will be given" must decide the
    quiz with the first clause and the other components with the second. In a Make-up section, a clause
    without a subject ("To be granted only in case of serious illness.") gets "Make-up:" in front so
    the model knows what is being granted.
    """
    from handout_models import Sentence
    clauses = []
    for sentence in sentences:
        for part in re.split(CLAUSE_SPLIT, sentence.text, flags=re.I):
            part = clean(part).strip(" ,.")
            if len(part.split()) < 3:
                continue
            # why: only subject-less clauses; "A student short of attendance shall not be permitted ..." has its own subject.
            if from_section and not re.search(MAKEUP_WORD, part, re.I) and re.match(r"(to be|will be|shall be|can be|may be|granted|allowed|given|permitted|only (?!those)|it (?:is solely|solely|depends))\b", part, re.I):
                part = f"Make-up: {part}"
            clauses.append(Sentence(part, sentence.page))
    return clauses


def only_for_exclusion(clauses, kind: str):
    """A make-up clause restricted to other named components ("only for Mid-Semester or Comprehensive"),
    or None. The kind is excluded when such a clause exists and names components but not this one."""
    alternatives = "|".join(COMPONENT_KINDS.values())
    for clause in clauses:
        lowered = clause.text.lower()
        if re.search(r"make\s*-?\s*ups?", lowered) and re.search(MAKEUP_ONLY_FOR.format(alternatives), lowered) \
                and not re.search(COMPONENT_KINDS.get(kind, r"(?!)"), lowered) and not re.search(r"\bno\b|\bnot\b", lowered):
            return clause
    return None


def makeup_premise_groups(clauses, kind: str, own_words: list[str] | None = None) -> list[tuple[list, list[tuple[str, bool | None, bool | None]]]]:
    """(premises, hypotheses) to try for one kind, most specific first.

    why: only a clause naming this component may decide it with the component hypothesis. A clause
    naming only other components ("no make-up for quizzes") is never a premise: the small NLI model reads
    it as a contradiction for every component. "Other components" clauses apply only to kinds that no
    clause names, and general clauses (naming nothing) come last. A general premise must mention make-up:
    any other sentence ("keep your answer sheets") contradicts "Make-up is never allowed." at random.
    """
    label = KIND_LABELS[kind]
    pattern = COMPONENT_KINDS.get(kind, r"(?!)")
    if own_words:  # the component's own name ("Internal tests" -> "test"), for "other" components
        pattern = rf"{pattern}|\b(?:{'|'.join(map(re.escape, own_words))})"
    # why: a naming clause must be about make-up ("Average marks of both quizzes will be considered" isn't).
    naming = [c for c in clauses if re.search(pattern, c.text.lower()) and re.search(MAKEUP_WORD, c.text, re.I)]
    if kind not in ("midsem", "compre"):  # "continuous evaluation" = everything except midsem and compre
        naming += [c for c in clauses if re.search(r"continuous evaluation", c.text, re.I) and re.search(MAKEUP_WORD, c.text, re.I)
                   and c not in naming]
    general = [c for c in clauses if not names_a_component(c.text) and re.search(MAKEUP_WORD, c.text, re.I)]
    others = [c for c in general if re.search(OTHER_COMPONENTS, c.text, re.I)]
    general = [c for c in general if c not in others]
    groups = [(naming, [(h.format(label=label), yes, no) for h, yes, no in HYPOTHESES["makeup.component"]])]
    if kind == "class_participation":
        # why: general make-up rules are about tests and exams; participation is only decided when named.
        return [(naming, groups[0][1])]
    if not naming:
        groups.append((others, HYPOTHESES["makeup.other_components"]))
    groups.append((general, HYPOTHESES["makeup.general"]))
    return [(premises, hypotheses) for premises, hypotheses in groups if premises] if naming else [([], [])] + [
        (premises, hypotheses) for premises, hypotheses in groups[1:] if premises]


def decide_makeup_kind(models, clauses, kind: str, stats: Counter, own_words: list[str] | None = None) -> tuple[bool | None, dict, list[str]]:
    """(value, method entry, issues) for one evaluation kind: premise groups in order, the first confident
    NLI decision wins; the "only for <other components>" exclusion (rule) sits after the naming group."""
    from handout_models import Decision
    groups = makeup_premise_groups(clauses, kind, own_words)
    excluded = only_for_exclusion(clauses, kind)
    if not any(premises for premises, _ in groups) and not excluded:
        stats["makeup.per_component:not_mentioned"] += 1
        return None, method_entry(Decision(None, None, None)), []
    query = FIELD_DESCRIPTIONS["makeup_component"].format(label=KIND_LABELS[kind])
    best = Decision(None, 0.0, None)
    # groups[0] is always the naming group (possibly empty), so the exclusion comes right after it.
    for index, (premises, hypotheses) in enumerate(groups):
        top = [item.sentence for item in models.locate(premises, query, TOP_K_SENTENCES)] if premises else []
        decision = models.decide(top, hypotheses)
        decision.neutral_confidence = max(decision.neutral_confidence, best.neutral_confidence)
        if (decision.confidence or 0) >= (best.confidence or 0):
            best = decision
        else:
            best.neutral_confidence = decision.neutral_confidence
        if (best.confidence or 0) >= NLI_MIN_CONFIDENCE:
            break
        if index == 0 and excluded:
            stats["makeup.per_component:decided"] += 1
            return False, method_entry(Decision(False, 1.0, excluded), method="rule: only-for exclusion"), []
    value, entry, issues = settle(best, "makeup.per_component", stats)
    return value, entry, [f"makeup_{kind}_uncertain" for _ in issues]


def extract_makeup(models, sentences, from_section: bool, kinds: list[str], stats: Counter,
                   own_words: dict[str, list[str]] | None = None) -> tuple[dict, dict, int | None, list[str]]:
    """(makeup, method entries per kind, page, issues). One decision per evaluation kind present.

    With a Make-up section, all its sentences are candidates once the best one passes the similarity
    threshold (a generic "granted only to genuine cases" can score low but is still the policy).
    Per kind: clauses naming it (NLI) -> "only for <others>" exclusion (rule) -> general clauses (NLI).
    """
    from handout_models import Decision
    located = models.locate(sentences, FIELD_DESCRIPTIONS["makeup"], len(sentences))
    mentioned = bool(located) and located[0].similarity >= MIN_SIMILARITY["makeup"]
    located_any = [item for item in located if mentioned and (from_section or item.similarity >= MIN_SIMILARITY["makeup"])]
    ordered = sorted(located_any, key=lambda item: sentences.index(item.sentence))
    text = " ".join(item.sentence.text for item in ordered) or None
    per_component: dict[str, bool | None] = {}
    methods: dict[str, dict] = {}
    issues: list[str] = []
    page: int | None = ordered[0].sentence.page if ordered else None
    clauses = makeup_clauses([item.sentence for item in ordered], from_section)
    own_words = own_words or {}
    for kind in kinds:
        per_component[kind], methods[kind], kind_issues = decide_makeup_kind(models, clauses, kind, stats, own_words.get(kind))
        issues += kind_issues
    stated = [value for value in per_component.values() if value is not None]
    if per_component.get("midsem") is True or per_component.get("compre") is True:
        allowed = True
    elif stated and all(value is False for value in stated):
        allowed = False
    else:
        allowed = None
    return {"per_component": per_component, "allowed": allowed, "text": text}, methods, page, issues


# ---------------------------------------------------------------- one handout

def title_matches(title: str | None, timetable_title: str | None) -> bool:
    """Does the handout title match the timetable's (abbreviated, upper-case) title for the file's code?

    'INTRO TO BIO SCIENCES' vs 'Introduction to Biological Sciences' -> True: every timetable word is a
    prefix of some title word. why: a printed code that differs from the file name is usually a cross-listing
    or an old/new code pair (BIO U101 printed as BIO F101); only a different title means a different course.
    """
    if not title or not timetable_title:
        return False
    title_words = re.findall(r"[a-z0-9]+", title.lower())
    wanted = [word for word in re.findall(r"[a-z0-9]+", timetable_title.lower()) if word not in ("and", "of", "the", "in", "for", "to")]
    hits = sum(any(word.startswith(part) for word in title_words) for part in wanted)
    return bool(wanted) and hits / len(wanted) >= TITLE_MATCH_SHARE


def course_plan_absent(sections: dict[str, list[Line]], title: str | None = None) -> bool:
    """True if the handout has no course plan to extract: project/thesis/study courses ("The plan of work will be
    decided by the respective instructors"), or a plan given only as a link."""
    plan = " ".join(line for _, line in sections.get("course_plan", []))
    if title and re.search(INDIVIDUAL_COURSE_TITLE, title, re.I):
        return True
    return len(plan.split()) < 60 and (not plan or bool(re.search(NO_PLAN_WORDING, plan, re.I)) or len(plan.split()) < 12)


def extract_handout(file_name: str, doc: Document, models, removed_log: Counter, stats: Counter,
                    timetable_titles: dict[str, str] | None = None) -> Handout:
    course_no = course_no_from_filename(Path(file_name)) or ""  # main() skips files without one
    issues: list[str] = []
    notes: list[str] = []  # facts about the source that were handled (not problems): see NOTE_TAGS
    sections = split_sections(doc.lines)
    block = header_block(doc.header_lines)

    printed, printed_page = header_value(block, NUMBER_LABEL)
    codes = printed_codes(printed or "")
    title, title_page = extract_title(block)
    if title is None:
        issues.append("title_not_found")
    if codes and course_no not in codes:
        # The timetable spells some codes with the suffix ("BITS F101-1"), so the raw file-name code is tried too.
        raw_code = re.sub(r"^\d+_([A-Z]+)_", r"\1 ", Path(file_name).stem)
        timetable_title = (timetable_titles or {}).get(course_no) or (timetable_titles or {}).get(raw_code)
        same_course = title_matches(title, timetable_title)
        (notes if same_course else issues).append(f"printed_code_{'differs' if same_course else 'mismatch'}: {'/'.join(codes)}")
    instructors, instructors_page = extract_instructors(block, doc.tables)
    about, about_page = build_about(sections, models, removed_log)
    if not about:  # the description is sometimes printed inside a table, which the masked lines leave out
        about, about_page = build_about(split_sections(doc.header_lines), models, removed_log)

    topics, topics_page, plan_table_found = extract_topics(doc.tables)
    if not topics:
        topics, topics_page = topics_from_label_tables(doc.tables)
        exact = True
        if not topics:
            topics, topics_page, exact = topics_from_text(sections.get("course_plan", []))
        if topics:  # no ruled plan table; read from the printed layout
            (notes if exact else issues).append("topics_from_text_layout" if exact else "topics_from_unruled_columns")
    if plan_table_found and len(topics) < TOPICS_MIN_EXPECTED:
        issues.append("topics_possibly_truncated")
    elif not topics:
        absent = course_plan_absent(split_sections(doc.header_lines), title)  # full text: plans often sit in tables
        (notes if absent else issues).append("no_course_plan_in_handout" if absent else "topics_not_found")

    evaluation, evaluation_page, evaluation_issues, evaluation_notes = extract_evaluation(doc, sections)
    evaluation_text = " ".join(line for _, line in split_sections(doc.header_lines).get("evaluation", []))
    if not evaluation and re.search(r"\bpass\b", evaluation_text, re.I) and re.search(r"\bfail\b", evaluation_text, re.I):
        evaluation_issues, evaluation_notes = [], ["pass_fail_course"]  # graded pass/fail: there are no weights
    issues += evaluation_issues
    notes += evaluation_notes

    attendance_section = sections.get("attendance")
    attendance, attendance_method, attendance_page, attendance_issues = extract_attendance(
        models, policy_sentences(attendance_section, doc, "attendance"),
        " ".join(line for _, line in attendance_section) if attendance_section else None, stats)
    kinds = list(dict.fromkeys(component.kind for component in evaluation))
    # "other" components are matched by the last word of their own name ("Internal tests" -> "test")
    own_words = {"other": sorted({re.sub(r"(e?s)$", "", words[-1]) for c in evaluation if c.kind == "other"
                                  and (words := re.findall(r"[a-z]{4,}", c.name.lower()))})}
    makeup, makeup_methods, makeup_page, makeup_issues = extract_makeup(
        models, policy_sentences(sections.get("makeup"), doc, "makeup"), bool(sections.get("makeup")), kinds, stats, own_words)
    issues += attendance_issues + makeup_issues

    consultation = sections.get("consultation", [])
    record = Handout(
        file=file_name,
        course_no=course_no,
        title=title,
        instructors=instructors,
        about=about,
        topics=topics,
        evaluation=evaluation,
        attendance=attendance,
        makeup=makeup,
        consultation_hours=" ".join(line for _, line in consultation) or None,
        extraction_methods={"attendance.required": attendance_method,
                            **{f"makeup.per_component.{kind}": entry for kind, entry in makeup_methods.items()}},
        source_pages={"file": None, "course_no": printed_page, "title": title_page, "instructors": instructors_page,
                      "about": about_page, "topics": topics_page, "evaluation": evaluation_page,
                      "attendance": attendance_page, "makeup": makeup_page,
                      "consultation_hours": consultation[0][0] if consultation else None},
        issues=issues,
        notes=notes,
    )
    if not about:
        record.issues.append("about_not_found")
    # why: notes are handled facts (cross-listed code, marks converted, no plan in a project course), not problems.
    record.needs_verification = bool(record.issues)
    return record


# ---------------------------------------------------------------- run + summary

def summarise(records: list[Handout], removed_log: Counter, stats: Counter) -> None:
    print(f"\nFiles: {len(records)}   unique course codes: {len({r.course_no for r in records})}")
    tags = Counter(re.sub(r"^(printed_code_mismatch|weightage_sum)_?:?.*", r"\1", tag) for r in records for tag in r.issues)
    print("Records with notes only (not flagged):", sum(bool(r.notes) and not r.issues for r in records))
    print("Issue tags:", dict(tags.most_common()))
    note_tags = Counter(re.sub(r"^(printed_code_differs):.*|^(best)_.*", lambda m: m[1] or "best_n_of_m", tag) for r in records for tag in r.notes)
    print("Notes (handled, not flagged):", dict(note_tags.most_common()))
    print("needs_verification = true:", sum(r.needs_verification for r in records))
    counts = sorted((len(r.topics), r.file) for r in records)
    print(f"Topics per handout: min {counts[0][0]}, median {statistics.median(c for c, _ in counts)}, max {counts[-1][0]}")
    print("Fewest topics:", counts[:10])
    required = Counter(r.attendance["required"] for r in records)
    print(f"attendance.required: true {required[True]}, false {required[False]}, null {required[None]} "
          f"(not mentioned {stats['attendance.required:not_mentioned']}, mentioned but not stated (neutral) "
          f"{stats['attendance.required:neutral']}, uncertain {stats['attendance.required:uncertain']})")
    makeup_values = Counter(value for r in records for value in r.makeup["per_component"].values())
    print(f"makeup.per_component: true {makeup_values[True]}, false {makeup_values[False]}, null {makeup_values[None]} "
          f"(not mentioned {stats['makeup.per_component:not_mentioned']}, neutral {stats['makeup.per_component:neutral']}, "
          f"uncertain {stats['makeup.per_component:uncertain']})")
    allowed = Counter(r.makeup["allowed"] for r in records)
    print(f"makeup.allowed (derived): true {allowed[True]}, false {allowed[False]}, null {allowed[None]}")
    print("Weightages converted from marks:", sum("weightage_in_marks_converted" in r.notes for r in records))
    print(f"Lines removed as headers while building 'about' ({len(removed_log)} distinct):")
    for line, count in removed_log.most_common():
        print(f"   {count:3} × {line}")


def load_timetable_titles() -> dict[str, str]:
    """course_no -> title from the processed timetable ({} if it hasn't been built yet)."""
    try:
        courses = json.loads(TIMETABLE_PATH.read_text())["courses"]
    except FileNotFoundError:
        log.warning("%s not found: printed-code mismatches can't be checked against titles", TIMETABLE_PATH)
        return {}
    except (ValueError, KeyError, TypeError) as error:
        log.warning("Could not read %s (%s)", TIMETABLE_PATH, error)
        return {}
    return {course["course_no"]: course["title"] for course in courses if course.get("course_no") and course.get("title")}


def load_pdf_safely(pdf_file: Path) -> tuple[str, Document | None]:
    """(file name, Document or None). Runs in a worker process; errors are logged, never raised."""
    try:
        return pdf_file.name, load_pdf(pdf_file)
    except Exception:
        log.exception("Could not parse %s", pdf_file.name)
        return pdf_file.name, None


def load_documents(files: list[Path]) -> dict[str, Document]:
    """Parse PDFs in parallel. why: table detection dominates the run time (about 1 s per handout)."""
    workers = max(1, min(PDF_WORKERS, os.cpu_count() or 1))
    log.info("Parsing %d PDFs with %d worker processes", len(files), workers)
    with multiprocessing.Pool(workers) as pool:
        loaded = pool.map(load_pdf_safely, files, chunksize=4)
    return {name: doc for name, doc in loaded if doc is not None}


def main(selected: list[str]) -> None:
    from handout_models import Models  # why: import here so the rule-only helpers can be used without torch
    if not HANDOUTS_DIR.is_dir():
        raise SystemExit(f"Handouts folder not found: {HANDOUTS_DIR}")
    files = [HANDOUTS_DIR / name for name in selected] if selected else sorted(HANDOUTS_DIR.glob("*.pdf"))
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise SystemExit(f"Not found: {missing}")

    skipped = [path.name for path in files if course_no_from_filename(path) is None]
    for name in skipped:
        log.warning("File name not in NNN_DEPT_CODE form, skipped: %s", name)
    files = [path for path in files if path.name not in skipped]
    manual = load_manual_documents()
    documents = load_documents([path for path in files if path.name not in manual])
    documents.update({path.name: manual[path.name] for path in files if path.name in manual})

    # why: models are loaded after the worker processes are done (torch/MPS state shouldn't be forked).
    models = Models(EMBEDDING_MODEL, NLI_MODEL)
    timetable_titles = load_timetable_titles()
    removed_log: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    records: list[Handout] = []
    for number, pdf_file in enumerate(files, 1):
        doc = documents.get(pdf_file.name)
        if doc is None:
            continue  # already logged by load_documents
        try:
            if not doc.lines:
                log.warning("%s has no text layer and no manual transcription", pdf_file.name)
            records.append(extract_handout(pdf_file.name, doc, models, removed_log, stats, timetable_titles))
        except Exception:
            # why: one odd handout shouldn't stop the other 539; logged with its traceback.
            log.exception("Could not extract %s", pdf_file.name)
        if number % 100 == 0:
            log.info("extracted %d / %d", number, len(files))

    output = SAMPLE_OUTPUT_PATH if selected else OUTPUT_PATH
    if not selected and OUTPUT_PATH.exists():
        OUTPUT_PATH.replace(OLD_OUTPUT_PATH)  # keep the previous output for comparison
    output.write_text(json.dumps([asdict(record) for record in records], indent=2, ensure_ascii=False))
    log.info("Saved %d records to %s", len(records), output)
    summarise(records, removed_log, stats)


if __name__ == "__main__":
    main(sys.argv[1:])
