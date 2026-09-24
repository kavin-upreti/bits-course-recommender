import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber
from pdfplumber.pdf import PDF

from timetable import PROJECT_ROOT, clean

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("pdfminer").setLevel(logging.ERROR)  # why: some handouts spam harmless font warnings

HANDOUTS_DIR = PROJECT_ROOT / "dataset" / "raw" / "handouts"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "code processed" / "handouts.json"
TIMETABLE_PATH = PROJECT_ROOT / "dataset" / "code processed" / "timetable.json"
# Hand-typed records for handouts the code can't read (scanned PDFs); used instead of parsing.
MANUAL_PATH = PROJECT_ROOT / "dataset" / "manually processed" / "handouts.json"

NUMBER_LABEL = r"course\s*(?:no\.?|number|code)"
# "Course Title", "Course Name", "Name of the course", or the combined "Course Number & Title"
TITLE_LABEL = r"course\s*(?:no\.?|number)\s*&\s*title|course\s*(?:title|name)|name\s*of\s*the\s*course"

# Section key -> regex that must match the WHOLE heading text (the part before ':', or the whole line).
# why: fullmatch means a body sentence like "Evaluation will be based on..." is never mistaken for a heading.
# Sections are found by keyword, never by number (Evaluation is section 5, 6 or 7 in different handouts).
SECTION_HEADINGS = {
    "description": r"((general|catalog|course) )*description( of the course)?",
    "objectives": r"(course )?(scope|objectives?)( ?(and|&) ?(objectives?|scope))?( of the course)?",
    "learning_outcomes": r"(course )?(learning )?outcomes?",
    "textbooks": r"text ?books?( ?(and|&) ?(reference books?|references?))?",
    "references": r"references?( books?)?|reference books?|other references|lab(oratory)? manual",
    "course_plan": r"(course|lecture|lesson) ?plan|plan of work|lecture schedule|course (content|schedule)|(list of )?experiments?( list)?",
    "evaluation": r"evaluation( scheme| components?| pattern| plan)?",
    "attendance": r"attendance( policy| requirements?)?",
    "makeup": r"make ?-?up( policy)?",
    "consultation": r".*consultation.*|office hours?",
}
# "8. Notices:" etc. A numbered heading we don't store still ends the section above it.
NUMBERED_HEADING = re.compile(r"(?:\d{1,2}|[IVX]{1,4})\s*[.)]\s*([A-Za-z].*)")
PAGE_NUMBER = re.compile(r"page \d+( of \d+)?", re.I)

# Start of a new book / outcome entry: "TB:", "T1", "[R2]", "RB1:", "1.", "a)", or a bullet.
ENTRY_MARKER = re.compile(r"^(\[?(TB|RB)\s*\d*\]?\s*[:.)\-]?\s*|\[?[TR]\s*\d+\]?\s*[:.)\-]?\s*|\d{1,2}\s*[.)]\s*|[a-z]\s*[.)]\s+|[•▪●○\-*]\s*)")
REFERENCE_MARKER = re.compile(r"^\[?(RB\s*\d*|R\s*\d+)\]?\s*[:.)\-]?", re.I)
SUBHEADING = re.compile(r"(text ?books?|reference books?|references|lab(?:oratory)? manual)\s*(\(.*?\))?\s*(:?)\s*(.*)", re.I)

# Evaluation component name -> kind. First match wins, so "Lab Viva" is a viva, not a lab.
COMPONENT_KINDS = {
    "midsem": r"mid",
    "compre": r"compre|final|end ?-?sem",
    "viva": r"viva",
    "lab": r"lab|practical|experiment",
    "quiz": r"quiz",
    "assignment": r"assign|home ?work",
    "project": r"project",
    "presentation": r"presentation|seminar",
    "tutorial": r"tut",
}
WEIGHT_TOLERANCE = 1  # percent; weights summing to 99-101 are accepted as complete
# why: the default (3) glues words together in PDFs with tight letter spacing ("additiontopartI").
X_TOLERANCE = 1.5
EVALUATION_HEADER = re.compile(r"weigh|marks|%")
PLAN_HEADER = re.compile(r"topic|modul|lec|week|session|content|chapter|descriptor|coverage")
# Plan column holding the topics, best first. Learning-outcome columns are never used as topics.
TOPIC_COLUMNS = [r"topic", r"description|descriptor|details|content|coverage", r"lecture session|session", r"module (title|name)|^modules?$"]

ATTENDANCE_REQUIRED = re.compile(r"mandatory|compulsory|must attend|required to attend|expected to attend|shall attend", re.I)
ATTENDANCE_OPTIONAL = re.compile(r"not (mandatory|compulsory)|optional|no attendance", re.I)
# why: \s* everywhere because some PDFs drop spaces ("Makeupwillbepermitted").
MAKEUP_MENTION = re.compile(r"make\s*-?\s*ups?", re.I)
MAKEUP_PERMISSIVE = re.compile(r"(given|granted|allowed|conducted|considered|held|permitted|applicable|provision|apply|request)", re.I)
MAKEUP_NEGATED = re.compile(r"\bno\s*make|\bnot\s*(be\s*)?(given|granted|allowed|permitted|conducted|held)", re.I)
MINOR_COMPONENTS = re.compile(r"quiz|assign|tut|lab|class|surprise|evaluative test|seminar|project", re.I)

Line = tuple[int, str]  # (page number, text)
Row = list[str | None]


@dataclass
class Component:
    name: str
    kind: str  # one of COMPONENT_KINDS, or "other"
    weightage_percent: float | None
    duration: str | None
    date: str | None
    nature: str | None  # "CB" closed book, "OB" open book, "CB/OB" both, None if not stated
    remarks: str | None


@dataclass
class Handout:
    file: str
    course_no: str  # from the file name, the same key as timetable.json
    title: str | None  # full title; the timetable's is abbreviated ("INTRO TO BIO SCI")
    description: str | None  # course description + scope and objective
    learning_outcomes: list[str] | None
    topics: list[str] | None  # from the course plan table
    textbooks: list[str] | None
    references: list[str] | None
    evaluation: list[Component]
    has_midsem: bool | None  # None = evaluation scheme couldn't be read
    attendance: dict  # {text, required: bool|None, percent: int|None}
    makeup: dict  # {text, allowed: bool|None}
    consultation_hours: str | None
    printed_course_no: str | None = None  # as printed inside the handout, e.g. "EEE / ECE / INSTR / CS F342"
    source_pages: dict[str, int] = field(default_factory=dict)  # field -> page it was read from
    needs_verification: bool = False
    note: str | None = None  # which fields failed a check and why


def course_no_from_filename(pdf_file: Path) -> str | None:
    """'002_BIO_F101.pdf' -> 'BIO F101'. A '-1' style suffix is kept only if the timetable uses it (see main)."""
    match = re.fullmatch(r"\d+_([A-Z]+)_(\S+)", pdf_file.stem)
    return f"{match[1]} {match[2]}" if match else None


def page_lines(pdf: PDF) -> list[Line]:
    """All text lines with their page number, minus repeated page headers/footers.

    why: every page repeats the institute letterhead; left in, it would be glued into
    whatever section spans the page break. A line counts as a header/footer if it sits in
    the top 4 or bottom 3 lines of at least two pages.
    """
    # why: dedupe_chars() undoes "fake bold" (each letter drawn twice), which otherwise reads as "BBIIOO FF221144".
    pages = [(page.page_number, (page.dedupe_chars().extract_text(x_tolerance=X_TOLERANCE) or "").splitlines()) for page in pdf.pages]
    edge_counts: dict[str, int] = {}
    for _, lines in pages:
        for line in set(lines[:4] + lines[-3:]):
            edge_counts[line] = edge_counts.get(line, 0) + 1
    boilerplate = {line for line, count in edge_counts.items() if count >= 2}

    return [
        (number, clean(line))
        for number, lines in pages
        for line in lines
        if clean(line) and line not in boilerplate and not PAGE_NUMBER.fullmatch(clean(line))
    ]


def match_heading(line: str) -> tuple[str | None, str] | None:
    """Return (section key or None for an unstored heading, text after the colon), or None if not a heading."""
    numbered = NUMBERED_HEADING.fullmatch(line)
    text = numbered[1] if numbered else line
    heading, colon, rest = text.partition(":")
    # why: drop "(TB)", "(if any)" and similar so "Text Book (TB)" matches "text ?books?".
    heading = clean(re.sub(r"\(.*?\)", "", heading)).rstrip(" .-–").lower()

    for key, pattern in SECTION_HEADINGS.items():
        if re.fullmatch(pattern, heading):
            return key, rest.strip()
    # why: an unknown numbered heading ("8. Notices:") still closes the previous section.
    # A short line with a colon, or a short line with no full stop, looks like a heading;
    # a numbered sentence inside a list ("1. To learn about probability.") does not.
    if numbered and len(heading) <= 50 and (colon or not heading.endswith(".")):
        return None, ""
    # Edge case (e.g. 091_CE_G527): "1 Course Description:" is numbered without a dot. Only known
    # headings are accepted this way, so table rows like "1 Introduction" can't end a section.
    bare = re.fullmatch(r"\d{1,2}\s+([A-Za-z].*)", line)
    if bare and not numbered:
        heading, _, rest = bare[1].partition(":")
        heading = clean(re.sub(r"\(.*?\)", "", heading)).rstrip(" .-–").lower()
        for key, pattern in SECTION_HEADINGS.items():
            if re.fullmatch(pattern, heading):
                return key, rest.strip()
    return None


def split_sections(lines: list[Line]) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Group lines under the heading above them. Returns (key -> lines, key -> page)."""
    sections: dict[str, list[str]] = {}
    pages: dict[str, int] = {}
    current: str | None = None

    for page_number, line in lines:
        heading = match_heading(line)
        if heading is None:
            if current:
                sections[current].append(line)
            continue
        current, rest = heading
        if current is None:
            continue
        # why: "Text Books" then "Reference Books" under one heading is common, but a second
        # "Evaluation:" inside notes shouldn't restart the section; the first occurrence wins.
        if current in sections:
            current = None
            continue
        sections[current] = [rest] if rest else []
        pages[current] = page_number

    return {key: body for key, body in sections.items() if body}, pages


def header_value(lines: list[Line], label: str) -> str | None:
    """Value after 'label :' in the handout's top block, e.g. 'Course Title : SURVEYING'."""
    for _, line in lines[:25]:
        # why: search, not match, because some put a prefix first ("1. a) Course Number: CS F446").
        match = re.search(rf"\b(?:{label})\s*:?\s*(.+)", line, re.I)
        if match:
            return match[1].strip(" []") or None
    return None


def printed_codes(printed: str) -> list[str]:
    """Course codes in a printed course-number line, tolerant of missing spaces.

    'GSF213' -> ['GS F213']; 'EEE / ECE / INSTR / CS F342' -> 4 codes; 'CS/SS G 527' -> ['CS G527', 'SS G527'].
    """
    printed = printed.upper()
    codes: list[str] = []
    for shared in re.finditer(r"((?:[A-Z]{2,5}\s*/\s*)+[A-Z]{2,5})\s+([A-Z])\s?(\d{3}[A-Z]?)", printed):
        codes += [f"{department.strip()} {shared[2]}{shared[3]}" for department in shared[1].split("/")]
    codes += [f"{match[1]} {match[2]}{match[3]}" for match in re.finditer(r"\b([A-Z]{2,5}?)\s*([A-Z])\s?(\d{3}[A-Z]?)\b", printed)]
    return list(dict.fromkeys(codes))


def title_similarity(handout_title: str, timetable_title: str) -> float:
    """Share of the timetable title's words found in the handout title.

    why: timetable titles are abbreviated ("ADV & APPLIED MICROBIO"), so a word matches if
    one is a prefix of the other ("MICROBIO" ~ "MICROBIOLOGY").
    """
    words = lambda text: [word for word in re.findall(r"[a-z]+", text.lower()) if len(word) > 2]
    ours, theirs = words(handout_title), words(timetable_title)
    if not theirs:
        return 0.0
    return sum(any(a.startswith(b) or b.startswith(a) for a in ours) for b in theirs) / len(theirs)


def choose_course_no(file_code: str, printed: str | None, title: str | None, timetable_titles: dict[str, str]) -> tuple[str, str | None]:
    """The course number to store, always one that exists in timetable.json when possible, plus a note.

    why: some handouts print a different number from their file name (cross-listing, typos, or the
    wrong handout uploaded). Candidates are the file-name number and every printed number that is in
    the timetable; the one whose timetable title best matches the handout's title wins, ties going
    to the file name.
    """
    if printed is None or printed_number_matches(file_code, printed) or file_code in printed_codes(printed):
        return file_code, None
    candidates = [file_code] + [code for code in printed_codes(printed) if code in timetable_titles and code != file_code]
    if len(candidates) == 1 or not title:
        return file_code, f"printed number {printed!r} differs; kept the file-name number"
    best = max(candidates, key=lambda code: (title_similarity(title, timetable_titles.get(code, "")), code == file_code))
    if best == file_code:
        return file_code, f"printed number {printed!r} differs; file-name number's timetable title matches best"
    return best, f"file name says {file_code} but the handout is for {best} (printed {printed!r}; timetable title matches)"


def printed_number_matches(course_no: str, printed: str) -> bool:
    """True if the handout's printed course number covers ours.

    why: cross-listed handouts print numbers in many shapes ("BITS F415/ BITS U415",
    "EEE / INSTR/ECE F366, F367", "ENGL C261 GS F241"), so instead of parsing them we only
    require our department and our number to each appear as a whole word.
    """
    department, number = course_no.split(" ", 1)
    printed = re.sub(r"\b([A-Z])\s+(\d{3})", r"\1\2", printed.upper())  # "BITS F 429" -> "BITS F429"
    return all(re.search(rf"(?<![A-Z0-9]){re.escape(part)}(?![A-Z0-9])", printed) for part in (department, number))


# ---------- lists (books, outcomes) ----------

def split_entries(lines: list[str]) -> list[str]:
    """Join wrapped lines into entries; a line starting with a marker (TB:, R1, 1., a), •) starts a new one."""
    entries: list[str] = []
    for line in lines:
        if not entries or ENTRY_MARKER.match(line):
            entries.append(line)
        else:
            entries[-1] = f"{entries[-1]} {line}"
    return [text for entry in entries if (text := ENTRY_MARKER.sub("", entry).strip(" :;"))]


def split_books(sections: dict[str, list[str]]) -> tuple[list[str] | None, list[str] | None]:
    """Textbooks and references as lists.

    why: many handouts put both under one heading ("Text Books and References"), either with
    a 'Reference Books:' sub-heading or with R1/RB markers, so each entry is routed by those.
    """
    textbooks: list[str] = []
    references: list[str] = []
    for key in ("textbooks", "references"):
        target = references if key == "references" else textbooks
        raw_entries: list[tuple[list[str], str]] = []
        for line in sections.get(key, []):
            sub = SUBHEADING.match(line)
            if sub and (len(line) < 30 or ":" in line[:30]):
                target = textbooks if sub[1].lower().startswith("text") else references
                if sub[4]:
                    raw_entries.append((target, sub[4]))
                continue
            if not raw_entries or ENTRY_MARKER.match(line):
                raw_entries.append((references if REFERENCE_MARKER.match(line) else target, line))
            else:
                raw_entries[-1] = (raw_entries[-1][0], f"{raw_entries[-1][1]} {line}")
        for destination, raw in raw_entries:
            text = ENTRY_MARKER.sub("", raw).strip(" :;")
            if text:
                destination.append(text)
    return textbooks or None, references or None


# ---------- tables (evaluation, course plan) ----------

def column_index(names: list[str], pattern: str, exclude: tuple[int | None, ...] = ()) -> int | None:
    return next((i for i, name in enumerate(names) if i not in exclude and re.search(pattern, name)), None)


def find_table(pdf: PDF, is_header: Callable[[list[str]], bool]) -> tuple[list[Row], int] | None:
    """Rows of the first table whose header passes is_header (header row first), and its page.

    why: tables often break across a page; headerless tables right after it with the
    same number of columns are its continuation.
    """
    rows: list[Row] = []
    found_page = 0
    for page in pdf.pages:
        for table in page.dedupe_chars().extract_tables({"text_x_tolerance": X_TOLERANCE}):
            if not table:
                continue
            header = is_header([clean(cell).lower() for cell in table[0]])
            if not rows and header:
                rows, found_page = list(table), page.page_number
            # why: some headers are a one-row table of their own; the next same-width table is then
            # the body even if its first row happens to contain a header word like "lecture".
            elif rows and len(table[0]) == len(rows[0]) and (not header or len(rows) == 1):
                rows.extend(table)
            elif rows:
                return rows, found_page
    return (rows, found_page) if rows else None


def is_evaluation_header(names: list[str]) -> bool:
    return column_index(names, EVALUATION_HEADER.pattern) is not None and column_index(
        names, r"component|evaluation|duration|date") is not None


def is_plan_header(names: list[str]) -> bool:
    return any(PLAN_HEADER.search(name) for name in names) and not any(EVALUATION_HEADER.search(name) for name in names)


def neighbour_cell(names: list[str], row: Row, index: int | None) -> str:
    """The row's value under a header; falls back to a blank-headed neighbour.

    why: merged header cells shift names off their data (header "Weightage" in column 3,
    the numbers in column 2 under an unnamed header). A neighbour is only used when its
    own header is blank, so a value is never taken from another named column.
    """
    if index is None:
        return ""
    for candidate in (index, index - 1, index + 1):
        if 0 <= candidate < len(row) and (candidate == index or not names[candidate]):
            value = clean(row[candidate])
            if value:
                return value
    return ""


def parse_weight(text: str) -> float | None:
    """'25%', '25', '50 (25 %)' -> 25.0 / 25.0 / 50.0 (a marks value; rescaled later)."""
    percent = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    number = percent or re.search(r"\d+(?:\.\d+)?", text)
    return float(number[1] if percent else number[0]) if number else None


def component_kind(name: str) -> str:
    lowered = name.lower()
    return next((kind for kind, pattern in COMPONENT_KINDS.items() if re.search(pattern, lowered)), "other")


def book_nature(text: str) -> str | None:
    """'Closed Book' -> 'CB', 'OB' -> 'OB', 'Closed & Open Book' -> 'CB/OB'."""
    closed = re.search(r"closed|close book|\bCB\b", text, re.I)
    opened = re.search(r"open|\bOB\b", text, re.I)
    if closed and opened:
        return "CB/OB"
    return "CB" if closed else "OB" if opened else None


def make_component(name: str, weight: float | None, duration: str, date: str, remarks: str) -> Component:
    return Component(
        name=name,
        kind=component_kind(name),
        weightage_percent=weight,
        duration=duration or None,
        date=date or None,
        nature=book_nature(f"{name} {remarks}"),
        remarks=remarks or None,
    )


def parse_evaluation_table(rows: list[Row]) -> list[Component]:
    """Turn the raw evaluation table into components. Columns are found by header name."""
    names = [clean(cell).lower() for cell in rows[0]]
    serial = tuple(i for i, name in enumerate(names) if re.fullmatch(r"#|s\.? ?no\.?|sl\.? ?no\.?|sr\.? ?no\.?|no\.?", name))
    weight_col = column_index(names, r"weigh|%|wt")
    if weight_col is None:
        weight_col = column_index(names, r"marks")
    # why: some tables label the name column, some leave it blank; fall back to the first real column.
    name_col = column_index(names, r"component|evaluation|test|type", (*serial, weight_col))
    if name_col is None:
        name_col = next((i for i in range(len(names)) if i not in (*serial, weight_col)), None)
    if name_col is None:
        return []
    duration_col = column_index(names, r"duration", (name_col, weight_col))
    date_col = column_index(names, r"date", (name_col, weight_col))
    remarks_col = column_index(names, r"remark|comment|nature|mode", (name_col, weight_col))

    components: list[Component] = []
    for row in rows[1:]:
        name = neighbour_cell(names, row, name_col)
        if not name or re.match(r"total", name, re.I):
            continue
        weight = parse_weight(neighbour_cell(names, row, weight_col))
        duration, date = neighbour_cell(names, row, duration_col), neighbour_cell(names, row, date_col)
        if weight is None and components and not (duration or date):
            # why: a row with only text is a wrapped line or footnote of the row above, not a component.
            previous = components[-1]
            if name[0].isalpha():
                previous.name = f"{previous.name} {name}"
                previous.kind = component_kind(previous.name)
            else:
                previous.remarks = f"{previous.remarks or ''} {name}".strip()
            continue
        components.append(make_component(name, weight, duration, date, neighbour_cell(names, row, remarks_col)))
    return components


def parse_evaluation_text(lines: list[str]) -> list[Component]:
    """Fallback for handouts whose evaluation 'table' has no ruling lines: one component per line with a weight.

    Handles 'Mid Semester Exam 20 90 min', 'Quiz 15% TBA' and '● 10% Class Participation'.
    The result is only trusted if the weights add up (checked by weights_are_complete).
    """
    components: list[Component] = []
    for line in lines:
        line = line.replace("[", "").replace("]", "")  # why: some handouts bracket every cell ("[Mid Semester] [30%]")
        text = re.sub(r"^[\W\d]{0,4}?(?=[A-Za-z%\d])", "", line) if not re.match(r"^\W*\d+\s*%", line) else line
        percent_first = re.match(r"^\W*(\d+(?:\.\d+)?)\s*%\s*([A-Za-z][^:]*)", text)
        percent_after = re.match(r"^([A-Za-z][^%]*?)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%", text)
        name_first = re.match(r"^([A-Za-z][A-Za-z &/\-().,']*?)\s*[:\-]?\s+(\d+(?:\.\d+)?)\b", text)
        if percent_first:
            weight, name = float(percent_first[1]), percent_first[2]
        elif percent_after:  # why: with a '%', trust that number ("Tutorial 1 5%" is 5%, not 1)
            name, weight = percent_after[1], float(percent_after[2])
        elif name_first:
            name, weight = name_first[1], float(name_first[2])
        else:
            continue
        name = re.sub(r"^\(?([a-h]|[ivx]{1,4})\)\s*", "", clean(name)).strip(" :-")  # "a) Final Report" -> "Final Report"
        # why: long names are sentences ("NC report ... below 30% of the average"), not components.
        # A lowercase start is a wrapped sentence ("top three performers or 40% of median"), not a component.
        if not name or name[0].islower() or re.match(r"(total|component|evaluation)\b", name, re.I) \
                or weight > 100 or len(name.split()) > 8:
            continue
        components.append(make_component(name, weight, "", "", text))
    return components


def weights_are_complete(components: list[Component]) -> bool:
    """True if weights add up to 100%, rescaling first when the table gave marks instead of percents.

    why: '(25 %)' style cells are already percents; a plain marks column (e.g. out of 200)
    is converted so every component ends up as a percent of the grade.
    """
    weights = [component.weightage_percent for component in components if component.weightage_percent is not None]
    if not weights or len(weights) != len(components):
        return False
    total = sum(weights)
    if abs(total - 100) <= WEIGHT_TOLERANCE:
        return True
    # ponytail: assumes any non-100 total is marks; a table missing one row will be rescaled wrongly,
    # so only accept it if the result is whole-ish percents. Upgrade: read the header's "marks" label.
    scaled = [round(weight * 100 / total, 1) for weight in weights]
    if total > 100 and all(abs(value - round(value)) < 0.05 for value in scaled):
        for component, value in zip(components, scaled):
            component.weightage_percent = value
        return True
    return False


def parse_topics(rows: list[Row]) -> list[str]:
    """Topic column of the course plan table, one entry per non-empty row."""
    names = [clean(cell).lower() for cell in rows[0]]
    topic_col = next((column_index(names, pattern) for pattern in TOPIC_COLUMNS if column_index(names, pattern) is not None), None)
    if topic_col is None:
        return []
    topics: list[str] = []
    for row in rows[1:]:
        topic = neighbour_cell(names, row, topic_col)
        if topic and not re.fullmatch(r"[\d\s.,\-–]+", topic) and topic not in topics:
            topics.append(topic)
    return topics


# ---------- policies ----------

def attendance_info(lines: list[str]) -> dict:
    """{text, required, percent}. None means the handout doesn't say (not 'no')."""
    text = "\n".join(lines) or None
    if text is None:
        return {"text": None, "required": None, "percent": None}
    # why: only a minimum-attendance percent counts; "attendance carries 10%" is a grade weightage.
    percent = re.search(r"(?:minimum|at\s*least|atleast|min\.?)\s*(?:of\s*)?(\d{2,3})\s*%|(\d{2,3})\s*%\s*(?:attendance|of\s*(?:the\s*)?(?:classes|lectures|sessions))", text, re.I)
    required = False if ATTENDANCE_OPTIONAL.search(text) else True if ATTENDANCE_REQUIRED.search(text) or percent else None
    return {"text": text, "required": required, "percent": int(percent[1] or percent[2]) if percent else None}


def makeup_info(lines: list[str]) -> dict:
    """{text, allowed}. allowed=True if a make-up is possible for some component (usually midsem/compre).

    why: handouts mix rules ("Make-up for midsem may be given...; no make-up for quizzes"), so each
    sentence is judged alone. A denial that only covers minor components (quizzes, tutorials) says
    nothing about the exams, so on its own it gives None, not False.
    """
    text = "\n".join(lines) or None
    if text is None:
        return {"text": None, "allowed": None}
    sentences = [sentence for sentence in re.split(r"(?<=[.;])\s+", " ".join(lines)) if MAKEUP_MENTION.search(sentence)]
    if any(MAKEUP_PERMISSIVE.search(sentence) and not MAKEUP_NEGATED.search(sentence) for sentence in sentences):
        return {"text": text, "allowed": True}
    denials = [sentence for sentence in sentences if MAKEUP_NEGATED.search(sentence)]
    if denials and not all(MINOR_COMPONENTS.search(sentence) for sentence in denials):
        return {"text": text, "allowed": False}
    return {"text": text, "allowed": None}


# ---------- one handout ----------

def parse_handout(pdf_file: Path, course_no: str, timetable_titles: dict[str, str]) -> Handout:
    """Extract every field for one handout file; anything that fails a check is flagged, not guessed."""
    with pdfplumber.open(pdf_file) as pdf:
        lines = page_lines(pdf)
        evaluation_table = find_table(pdf, is_evaluation_header)
        plan_table = find_table(pdf, is_plan_header)

    problems: list[str] = []
    if not lines:
        # why: a scanned handout has no text layer; OCR would be needed, so flag it instead.
        return Handout(file=pdf_file.name, course_no=course_no, title=None, description=None, learning_outcomes=None,
                       topics=None, textbooks=None, references=None, evaluation=[], has_midsem=None,
                       attendance=attendance_info([]), makeup=makeup_info([]), consultation_hours=None,
                       needs_verification=True, note="scanned PDF with no text layer (needs OCR or manual entry)")

    sections, pages = split_sections(lines)
    source_pages = {key: page for key, page in pages.items()}

    title = header_value(lines, TITLE_LABEL)
    if title:
        title = re.sub(r"^(?:[A-Z]{2,5}\s*/\s*)*[A-Z]{2,5}\s*[A-Z]\s?\d{3}[A-Z]?\s*", "", title).strip() or None  # combined "Number & Title"
    printed = header_value(lines, NUMBER_LABEL)
    course_no, number_note = choose_course_no(course_no, printed, title, timetable_titles)
    notes = [number_note] if number_note else []

    description = "\n".join(sections.get("description", []) + sections.get("objectives", [])) or None
    if description is None:
        problems.append("no description or scope/objective section found")
    if "objectives" in pages and "description" not in pages:
        source_pages["description"] = pages["objectives"]

    evaluation: list[Component] = []
    if evaluation_table:
        evaluation = parse_evaluation_table(evaluation_table[0])
        source_pages["evaluation"] = evaluation_table[1]
    if not weights_are_complete(evaluation):
        from_text = parse_evaluation_text(sections.get("evaluation", []))
        if weights_are_complete(from_text):
            evaluation = from_text
        else:
            problems.append("evaluation weights missing or don't add up to 100%")
    evaluation_ok = weights_are_complete(evaluation)
    has_midsem = any(component.kind == "midsem" for component in evaluation) if evaluation_ok else None

    topics = parse_topics(plan_table[0]) if plan_table else []
    if plan_table:
        source_pages["topics"] = plan_table[1]
    # why: project/thesis courses have no plan at all ("not mentioned" -> null, no flag);
    # a plan section that exists but couldn't be read is a real gap.
    # A plan written as prose ("The plan of work will be decided by the supervisor") has no topics.
    looks_like_schedule = sum(bool(re.match(r"^(L|Lec\.?|Week)?\s*\d+(\s*[-–]\s*\d+)?\b", line)) for line in sections.get("course_plan", [])) >= 3
    if not topics and looks_like_schedule:
        problems.append("course plan found but no table with a topic column")

    textbooks, references = split_books(sections)
    if title is None:
        problems.append("title not found in the header")

    return Handout(
        file=pdf_file.name,
        course_no=course_no,
        title=title,
        description=description,
        learning_outcomes=split_entries(sections.get("learning_outcomes", [])) or None,
        topics=topics or None,
        textbooks=textbooks,
        references=references,
        evaluation=evaluation,
        has_midsem=has_midsem,
        attendance=attendance_info(sections.get("attendance", [])),
        makeup=makeup_info(sections.get("makeup", [])),
        consultation_hours="\n".join(sections.get("consultation", [])) or None,
        printed_course_no=printed,
        source_pages=source_pages,
        needs_verification=bool(problems),
        note="; ".join(problems + notes) or None,
    )


def timetable_titles() -> dict[str, str]:
    """course_no -> title from timetable.json (run timetable.py first)."""
    try:
        return {course["course_no"]: course["title"] for course in json.loads(TIMETABLE_PATH.read_text())["courses"]}
    except (OSError, ValueError, KeyError, TypeError) as error:
        log.warning("Could not read %s (%s); course numbers won't be checked against it", TIMETABLE_PATH, error)
        return {}


def manual_records() -> dict[str, dict]:
    """file name -> hand-typed record from manually processed/handouts.json (empty if the file is absent)."""
    try:
        return {record["file"]: record for record in json.loads(MANUAL_PATH.read_text())["handouts"]}
    except FileNotFoundError:
        return {}
    except (ValueError, KeyError, TypeError) as error:
        log.warning("Could not read %s (%s); manual records ignored", MANUAL_PATH, error)
        return {}


def resolve_course_no(pdf_file: Path, known: set[str]) -> str | None:
    """Course number for a handout file; 'BITS F101-1' keeps its suffix only if the timetable uses it."""
    course_no = course_no_from_filename(pdf_file)
    if course_no is None:
        return None
    stripped = re.sub(r"-\d+$", "", course_no)
    return course_no if course_no in known or stripped == course_no else stripped


def extract_all(handouts_dir: Path) -> list[Handout]:
    """Parse every handout file (one record per file; a course may have several)."""
    titles = timetable_titles()
    manual = manual_records()
    handouts: list[Handout] = []
    for pdf_file in sorted(handouts_dir.glob("*.pdf")):
        course_no = resolve_course_no(pdf_file, set(titles))
        if course_no is None:
            log.warning("Handout file name not in NNN_DEPT_NUMBER form, skipped: %s", pdf_file.name)
            continue
        if pdf_file.name in manual:
            # why: scanned PDFs have no text; their fields were typed in by hand.
            record = dict(manual[pdf_file.name])
            record["evaluation"] = [Component(**component) for component in record["evaluation"]]
            handouts.append(Handout(**record))
            continue
        try:
            handouts.append(parse_handout(pdf_file, course_no, titles))
        except Exception:
            # why: one corrupt PDF shouldn't stop the other 539; it is logged with its traceback.
            log.exception("Could not parse %s", pdf_file.name)
    return handouts


def save(handouts: list[Handout], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(handout) for handout in handouts], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if not HANDOUTS_DIR.is_dir():
        raise SystemExit(f"Handouts folder not found: {HANDOUTS_DIR}")

    all_handouts = extract_all(HANDOUTS_DIR)
    save(all_handouts, OUTPUT_PATH)
    flagged = sum(handout.needs_verification for handout in all_handouts)
    log.info("Saved %d handouts (%d flagged for verification) to %s", len(all_handouts), flagged, OUTPUT_PATH)
