import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber
from pdfplumber.page import Page

from timetable import PROJECT_ROOT, clean

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

PDF_PATH = PROJECT_ROOT / "dataset" / "raw" / "bulletin.pdf"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "code processed" / "bulletin.json"

# Headings that mark where each part of the Bulletin starts (document format, not content).
PATTERN_TITLE = re.compile(r"(pattern \d+\s+)?semester-?\s?wise pattern", re.I)
COURSE_LISTS_START = "List of Courses for"
MINORS_START = re.compile(r"MINOR PROGRAMMES FOR FIRST")
DESCRIPTIONS_START = "Course Description for all On-campus"
DESCRIPTIONS_END = re.compile(r"^\s*PART VII", re.M)

# "CS F211", "BITS F421T", "BITS F 385", "MGTS F 211"
CODE = re.compile(r"\b([A-Z]{2,5})\s+([A-Z])\s?(\d{3}[A-Z]?)\b")
BARE_NUMBER = re.compile(r"^([A-Z])\s?(\d{3}[A-Z]?)\b")  # "F425T" after "or", department left out
UNITS_TAIL = re.compile(r"\s+((?:\d+|-)\s+(?:\d+|-)\s+\d+\*?|\d+\*?)\s*$")  # "3 0 3", "3", "4*"
PAGE_LABEL = re.compile(r"^([IVX]+\s*-\s*\d+|[IVX]|-?\d+|IV-)$")
OR_LINE = re.compile(r"^(or|OR|Or)(\s+(or|OR))?$")
ELECTIVE_SLOT = re.compile(
    r"(open\s*/\s*humanities|humanities\s*/\s*open|humanities|open|first discipline|second discipline|discipline)\s+electives?", re.I)
ELECTIVE_CATEGORIES = {"open/humanities": "Open/Humanities", "humanities/open": "Open/Humanities", "humanities": "HUEL",
                       "open": "OPEL", "discipline": "DEL", "first discipline": "DEL (first degree)",
                       "second discipline": "DEL (second degree)"}
FINAL_YEAR = re.compile(r"practice school\s*[-–]?\s*ii\b|thesis", re.I)
# "Discipline Core - 48 Units (16 Courses)", "Discipline Electives - 15 Units (min)-(4 Courses (min))"
FOOTER_CORE = re.compile(r"Discipline Core\s*[-–:]?\s*(\d+)\s*Units?\s*(?:\(min\))?\s*[-–]?\s*\(?\s*(\d+)\s*Courses?", re.I)
FOOTER_DEL = re.compile(r"Discipline Electives?\s*[-–:]?\s*(\d+)\s*Units?\s*(?:\(min\))?\s*[-–]?\s*\(?\s*(\d+)\s*Courses?", re.I)
FOOTER_LINE = re.compile(r"^\W*Discipline (Core|Electives?)\b.*\bUnits?\b.*$", re.I | re.M)
EXCLUSION = re.compile(r"[^.]*\b(exclusively|not (be )?(open|available|eligible|allowed|offered)|except|excluding|cannot|only for)\b[^.]*\.", re.I)
ROMAN_YEARS = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}


# ---------- small helpers ----------

def normalise_code(department: str, letter: str, number: str) -> str:
    return f"{department} {letter}{number}"


def find_codes(text: str) -> list[str]:
    """Every course code in the text, in order, e.g. 'CHE F243/ME F216 / MF F216' -> 3 codes.

    why: 'EEE/INSTR F432' shares one number between two departments, so it becomes EEE F432 + INSTR F432.
    """
    shared = re.sub(r"\b([A-Z]{2,5})\s*/\s*([A-Z]{2,5})\s+([A-Z]\s?\d{3}[A-Z]?)", r"\1 \3 / \2 \3", text)
    return [normalise_code(*match) for match in CODE.findall(shared)]


def split_units(text: str) -> tuple[str, dict[str, int | None]]:
    """'Operating Systems 3 0 3' -> ('Operating Systems', {L:3, P:0, U:3}); a lone number is U only."""
    match = UNITS_TAIL.search(text)
    if match is None:
        return text.strip(), {"L": None, "P": None, "U": None}
    numbers = match[1].split()
    as_int = [int(value.rstrip("*")) if value.rstrip("*").isdigit() else 0 for value in numbers]
    units = {"L": as_int[0], "P": as_int[1], "U": as_int[2]} if len(as_int) == 3 else {"L": None, "P": None, "U": as_int[0]}
    return text[: match.start()].strip(), units


def column_lines(page: Page) -> list[str]:
    """Text of a two-column page in reading order: left column, then right column."""
    middle = page.width / 2
    lines: list[str] = []
    for box in ((0, 0, middle, page.height), (middle, 0, page.width, page.height)):
        lines.extend(clean(line) for line in (page.crop(box).extract_text() or "").splitlines())
    return [line for line in lines if line and not PAGE_LABEL.fullmatch(line)]


def join_split_codes(lines: list[str]) -> list[str]:
    """'ECOM Real Time Operating Systems 3 1 4' + 'F321' -> 'ECOM F321 Real Time Operating Systems 3 1 4'.

    why: long department codes push the course number onto its own line in narrow columns.
    """
    joined: list[str] = []
    skip = False
    for index, line in enumerate(lines):
        if skip:
            skip = False
            continue
        following = lines[index + 1] if index + 1 < len(lines) else ""
        head = re.match(r"^([A-Z]{2,5})\s+(?![A-Z]\s?\d{3})(.+)$", line)
        if head and re.fullmatch(r"[A-Z]\d{3}[A-Z]?", following):
            joined.append(f"{head[1]} {following} {head[2]}")
            skip = True
        else:
            joined.append(line)
    return joined


def source(page_number: int, section: str) -> dict:
    return {"doc": PDF_PATH.name, "section": section, "page": page_number}


# ---------- data shapes ----------

@dataclass
class ListedCourse:
    code: str
    title: str
    L: int | None
    P: int | None
    U: int | None
    alternatives: list[str] = field(default_factory=list)
    pool: str | None = None  # track / pool name for discipline electives
    non_letter_grade: bool = False  # '*' in the Bulletin: graded GOOD/POOR


@dataclass
class CourseList:
    discipline: str | None  # None if the heading isn't in the PDF's text (see note)
    core: list[ListedCourse] = field(default_factory=list)
    discipline_electives: list[ListedCourse] = field(default_factory=list)
    project_courses: list[str] = field(default_factory=list)
    source: dict = field(default_factory=dict)
    needs_verification: bool = False
    note: str | None = None


@dataclass
class Slot:
    slot_type: str  # "named" or "elective"
    code: str | None = None
    title: str | None = None
    alternatives: list[str] = field(default_factory=list)
    category: str | None = None  # elective slots: HUEL / OPEL / DEL / Open/Humanities
    units_text: str | None = None  # as printed; unreliable for named courses (use course lists)


@dataclass
class SemesterChart:
    year: int
    semester: int
    slots: list[Slot] = field(default_factory=list)
    unit_total: str | None = None  # "18/21"
    same_as_first_degree: bool = False


@dataclass
class Programme:
    name: str
    type: str  # "single", "dual", or "dual_template" (the generic Pattern 1/2/3 charts)
    degree: str | None
    components: list[str]
    edition: str | None
    batch_range: None = None  # not stated in the Bulletin
    semesters: list[SemesterChart] = field(default_factory=list)
    summer: dict | None = None
    final_year_options: list[Slot] = field(default_factory=list)
    footer: dict = field(default_factory=dict)
    cdc_lists: list[str] = field(default_factory=list)  # discipline(s) in course_lists whose core = this programme's CDCs
    source: dict = field(default_factory=dict)
    needs_verification: bool = True  # always: batch_range is unknown
    note: str | None = "batch_range not stated in the Bulletin"


@dataclass
class Minor:
    name: str
    description: str | None = None
    min_courses: int | None = None
    min_units: int | None = None
    core: list[ListedCourse] = field(default_factory=list)
    electives: list[ListedCourse] = field(default_factory=list)
    exclusion_text: str | None = None
    source: dict = field(default_factory=dict)
    needs_verification: bool = False
    note: str | None = None


@dataclass
class CourseDescription:
    code: str
    title: str
    L: int | None
    P: int | None
    U: int | None
    department: str
    description: str | None
    prerequisites: list[list[str]] | None  # AND of OR-groups: [["CE F231", "ME F212"], ["MATH F211"]]
    prerequisites_text: str | None
    source: dict = field(default_factory=dict)
    needs_verification: bool = False
    note: str | None = None


def flag(record, reason: str) -> None:
    record.needs_verification = True
    record.note = f"{record.note}; {reason}" if record.note else reason


# ---------- (a) programme charts ----------

def programme_identity(title: str) -> tuple[str, str, str | None, list[str]]:
    """(name, type, degree, components) from a chart title."""
    title = re.sub(r"\bB\.\s*E\.", "B.E.", re.sub(r"\bM\.\s*Sc\.", "M.Sc.", clean(title)))
    if re.match(r"pattern \d", title, re.I):
        return title, "dual_template", None, []
    dual = re.search(r"\(?\s*((?:M\.Sc\.|B\.E\.|B\. ?Pharm\.)[^()]*?)\s+with\s+([^()]+?)\s*\)?$", title)
    if dual and re.search(r"dual", title, re.I):
        first, second = clean(dual[1]), clean(re.sub(r"\s*Programmes?$", "", dual[2]))
        return f"{first} with {second}", "dual", None, [first, second]
    single = re.search(r"admitted to\s+(.+?)(\s+programme)?$", title, re.I)
    name = clean(single[1]) if single else title
    degree = re.match(r"(B\.E\.|M\.Sc\.|B\. ?Pharm\.|Bachelor of [A-Za-z ]+)", name)
    return name, "single", degree[1] if degree else None, []


def parse_chart_cell(text: str, units_text: str) -> tuple[list[Slot], bool]:
    """Slots in one semester cell, and whether the printed units could be matched to them.

    why: a cell mixes wrapped titles, codes, 'or' lines and elective slots. Only codes and elective
    slots become slots; titles come from the course lists later. The U column has one line per slot
    (plus one per alternative), so units are attached only when the counts line up.
    """
    slots: list[Slot] = []
    pending_alternative = False
    for line in text.split("\n"):
        line = clean(line)
        if not line:
            continue
        if OR_LINE.fullmatch(line):
            pending_alternative = True
            continue
        or_prefix = re.match(r"^(or|OR|Or)\s+(.*)$", line)
        if or_prefix:
            pending_alternative, line = True, or_prefix[2]
        codes = find_codes(line)
        if not codes and pending_alternative and slots and slots[-1].code:
            bare = BARE_NUMBER.match(line)  # "F425T Thesis": department left out
            if bare:
                codes = [f"{slots[-1].code.split()[0]} {bare[1]}{bare[2]}"]
        if codes and CODE.match(line) or (codes and pending_alternative):
            title = clean(CODE.sub("", line, count=1)).rstrip(" or") or None
            if pending_alternative and slots and slots[-1].slot_type == "named":
                slots[-1].alternatives.append(codes[0])
            else:
                slots.append(Slot("named", code=codes[0], title=title))
            pending_alternative = False
            continue
        elective = ELECTIVE_SLOT.search(line)
        if elective:
            category = ELECTIVE_CATEGORIES[re.sub(r"\s+", " ", elective[1].lower().replace(" / ", "/"))]
            slots.append(Slot("elective", category=category))
            pending_alternative = False

    unit_lines = [clean(line) for line in units_text.split("\n") if clean(line) and not OR_LINE.fullmatch(clean(line))]
    expected = sum(1 + len(slot.alternatives) for slot in slots)
    if slots and len(unit_lines) == expected:
        position = 0
        for slot in slots:
            slot.units_text = unit_lines[position]
            position += 1 + len(slot.alternatives)
        return slots, True
    return slots, not slots


def parse_chart(table: list[list[str | None]], programme: Programme) -> None:
    """Fill programme.semesters / summer / final_year_options from one chart table (or its continuation)."""
    year: int | None = None
    charts = {(chart.year, chart.semester): chart for chart in programme.semesters}
    for row in table:
        cells = [(cell or "") for cell in row] + [""] * (5 - len(row))
        first = clean(cells[0])
        if PATTERN_TITLE.match(first) or first == "Year":
            continue
        if first.lower().startswith("summer"):
            programme.summer = {"codes": find_codes(first), "text": first}
            continue
        if first in ROMAN_YEARS:
            year = ROMAN_YEARS[first]
        if year is None:
            continue
        if clean(cells[1]) == "First Semester":  # dual charts repeat the header per year
            continue
        if not clean(cells[1]) and not clean(cells[3]):  # "18 | 19" totals row
            for semester, total in ((1, clean(cells[2])), (2, clean(cells[4]))):
                if total and (year, semester) in charts:
                    charts[(year, semester)].unit_total = total
            continue
        for semester, text, units in ((1, cells[1], cells[2]), (2, cells[3], cells[4])):
            chart = charts.setdefault((year, semester), SemesterChart(year, semester))
            if chart not in programme.semesters:
                programme.semesters.append(chart)
            if re.search(r"same as first degree", text, re.I):
                chart.same_as_first_degree = True
                continue
            # why: when printed units don't line up with the slots, units_text stays None; units are
            # taken from the course lists anyway (the Bulletin's per-line units are unreliable).
            slots, _ = parse_chart_cell(text, units)
            chart.slots.extend(slots)
            programme.final_year_options.extend(
                slot for slot in slots if slot.title and FINAL_YEAR.search(slot.title) and slot not in programme.final_year_options)


def parse_footer(page_text: str, programme: Programme) -> None:
    """'Discipline Core - 48 Units (16 Courses)' / 'Discipline Electives-12 Units (4 Courses)'."""
    core, electives = FOOTER_CORE.search(page_text), FOOTER_DEL.search(page_text)
    programme.footer = {
        "core_units": int(core[1]) if core else None,
        "core_courses": int(core[2]) if core else None,
        "del_units": int(electives[1]) if electives else None,
        "del_courses": int(electives[2]) if electives else None,
        "text": [clean(match.group(0)) for match in FOOTER_LINE.finditer(page_text)] or None,
    }
    if programme.type != "single":
        programme.footer = {}
        programme.note += "; no footer on dual-degree charts: each component programme's footer applies"
    elif core is None or electives is None:
        flag(programme, "footer missing or not in 'N Units (M Courses)' form; see footer.text")


def extract_programmes(pdf: pdfplumber.PDF, first_page: int, last_page: int, edition: str | None) -> list[Programme]:
    programmes: list[Programme] = []
    current: Programme | None = None
    for page in pdf.pages[first_page - 1: last_page]:
        tables = page.extract_tables()
        for table in tables:
            if not table or not table[0]:
                continue
            title = clean(table[0][0])
            if PATTERN_TITLE.match(title):
                name, kind, degree, components = programme_identity(title)
                current = Programme(name, kind, degree, components, edition, source=source(page.page_number, "Semester-wise pattern"))
                programmes.append(current)
                parse_footer(page.extract_text() or "", current)
            elif current is None:
                continue
            parse_chart(table, current)
    return programmes


# ---------- (b), (c) course lists, HUEL pool, audit courses ----------

def parse_course_lists(lines_by_page: list[tuple[int, list[str]]]) -> tuple[list[CourseList], list[ListedCourse], list[ListedCourse], list[ListedCourse]]:
    """Walk the 'List of Courses' pages as one stream of lines.

    Returns (discipline lists, HUEL pool, audit courses, project course patterns 'XXX F266').
    """
    lists: list[CourseList] = []
    huel_pool: list[ListedCourse] = []
    audit: list[ListedCourse] = []
    project_patterns: list[ListedCourse] = []
    current: CourseList | None = None
    section: str | None = None  # core / electives / project / huel / other / audit
    pool: str | None = None
    last: ListedCourse | None = None
    pending_alternative = False
    heading_parts: list[str] = []

    def target() -> list[ListedCourse] | None:
        if section == "core" and current:
            return current.core
        if section == "electives" and current:
            return current.discipline_electives
        return {"huel": huel_pool, "audit": audit, "project": project_patterns}.get(section or "")

    for page_number, lines in lines_by_page:
        for line in join_split_codes(lines):
            is_heading_word = line.isupper() and not CODE.match(line) and len(line) > 3 and not OR_LINE.fullmatch(line)
            if heading_parts and not (is_heading_word and not re.match(r"(CORE|DISCIPLINE ELECTIVE)", line)):
                # why: discipline headings wrap ("ELECTRICAL AND ELECTRONICS" / "ENGINEERING"); join then start the list.
                current = CourseList(" ".join(heading_parts), source=source(page_number, "List of Courses"))
                lists.append(current)
                heading_parts, section, pool, last = [], None, None, None

            if OR_LINE.fullmatch(line):
                pending_alternative = True
                continue
            if re.match(r"CORE COURSES", line):
                if current is not None and current.core:
                    # why: a second core list under one heading means the next discipline's heading
                    # isn't in the text layer (e.g. printed as an image); keep it, unnamed and flagged.
                    current = CourseList(None, source=source(page_number, "List of Courses"))
                    flag(current, "discipline heading not found in the PDF text")
                    lists.append(current)
                section, pool, last = "core", None, None
                continue
            if re.match(r"DISCIPLINE ELECTIVE COURSES", line):
                section, pool, last = "electives", None, None
                continue
            if re.match(r"Pool of Humanities", line, re.I):
                section, last, current = "huel", None, None
                continue
            if re.match(r"(Track|Pool)\b", line) and section == "electives":
                pool, last = line, None
                continue
            if re.fullmatch(r"Project Type Courses", line, re.I):
                section, last, current = "project", None, None
                continue
            if re.fullmatch(r"Other Courses", line, re.I):
                section, last = "other", None  # why: after the HUEL pool; these are NOT humanities electives
                continue
            if re.match(r"List of Audit Type Courses", line, re.I):
                section, last = "audit", None
                continue

            course = re.match(r"^\*?\s*([A-Z]{2,5})\s+([A-Z])\s?(\d{3}[A-Z]?)\s+(.*)$", line)
            if course:
                title, units = split_units(course[4])
                listed = ListedCourse(normalise_code(course[1], course[2], course[3]), title, **units, pool=pool,
                                      non_letter_grade=line.startswith("*") or bool(re.search(r"\d\*\s*$", line)))
                destination = target()
                if pending_alternative and last is not None:
                    last.alternatives.append(listed.code)
                elif destination is not None:
                    destination.append(listed)
                    last = listed
                pending_alternative = False
                continue

            if is_heading_word and section != "audit":
                heading_parts.append(line)
                continue
            # Wrapped title: a short line right after a course line.
            if last is not None and len(line) < 45 and not line.startswith("*") and ":" not in line:
                last.title = f"{last.title} {line}".strip()
            else:
                last = None

    for course_list in lists:
        if not course_list.core:
            flag(course_list, "no core courses found under this heading")
        departments = Counter(course.code.split()[0] for course in course_list.core)
        if course_list.discipline is None and departments:
            flag(course_list, f"most common core-course department is {departments.most_common(1)[0][0]}")
        if departments and project_patterns:
            prefix = departments.most_common(1)[0][0]
            course_list.project_courses = [pattern.code.replace("XXX", prefix) for pattern in project_patterns]
            # why: a note, not a flag: the XXX pattern is stated by the Bulletin, only the prefix is inferred.
            inferred = f"project course prefix '{prefix}' inferred as the most common core-course department"
            course_list.note = f"{course_list.note}; {inferred}" if course_list.note else inferred
    return lists, huel_pool, audit, project_patterns


# ---------- (d) minors ----------

def parse_minor_table(table: list[list[str | None]], minor: Minor, role: str | None) -> str | None:
    """Rows: Description / Courses & Units / a role label (merged cell, text on its first row) / courses.

    Returns the role in effect at the end, so a table continued on the next page keeps it.
    """
    pending_alternative = False
    for row in table:
        cells = [clean(cell) for cell in row] + [""] * (6 - len(row))
        label, value = cells[0], cells[1]
        if label.startswith("Minor in"):
            continue
        if re.match(r"core", label, re.I):
            role = "core"
        elif re.match(r"elective", label, re.I):
            role = "electives"
        text = " ".join(cells)
        requirement = re.search(r"(\d+)\s*courses?\s*\(min\)", text, re.I)
        if requirement and minor.min_courses is None:
            minor.min_courses = int(requirement[1])
            units = re.search(r"(\d+)\s*units?\s*\(min\)", text, re.I)
            minor.min_units = int(units[1]) if units else None
            continue
        if value.lower() == "or":
            pending_alternative = True
            continue
        codes = find_codes(value)
        if not codes:
            if (label == "Description" or (not label and minor.core == [] and minor.electives == [])) and value \
                    and not re.match(r"course (no|number)", value, re.I):
                minor.description = f"{minor.description} {value}".strip() if minor.description else value
            continue
        first_line = lambda index: (row[index] or "").split("\n")[0] if index < len(row) else ""
        units = [first_line(index).strip() for index in (3, 4, 5)]
        listed = ListedCourse(codes[0], cells[2], *[int(u.rstrip("*")) if u.rstrip("*").isdigit() else None for u in units],
                              alternatives=codes[1:], pool=label if role == "electives" and label else None,
                              non_letter_grade=units[2].endswith("*"))
        if pending_alternative:
            last = (minor.electives or minor.core)[-1] if (minor.electives or minor.core) else None
            if last:
                last.alternatives.append(listed.code)
            pending_alternative = False
            continue
        if role is None:
            flag(minor, f"{listed.code} listed before any Core/Electives label; role not assigned")
            continue
        (minor.core if role == "core" else minor.electives).append(listed)
    return role


def extract_minors(pdf: pdfplumber.PDF, start_page: int, stop_page: int) -> list[Minor]:
    minors: list[Minor] = []
    role: str | None = None
    titles_in_text: list[str] = []  # "Minor in X" lines on earlier pages, in order
    for page in pdf.pages[start_page - 1: stop_page]:
        for table in page.extract_tables():
            if not table or not table[0]:
                continue
            title = clean(table[0][0])
            name = title.removeprefix("Minor in").strip()
            used = {minor.name for minor in minors}
            # why: a minor continued on the next page repeats its title; that's not a new minor.
            if title.startswith("Minor in") and name not in used:
                minors.append(Minor(name, source=source(page.page_number, "Minor Programmes")))
                role = None
            elif title == "Description" and minors and minors[-1].description:
                # why: sometimes the title is printed above the table (end of the previous page),
                # so the table starts at "Description"; name it from the latest unused title in the text.
                unused = [text_title for text_title in titles_in_text if text_title not in used]
                minors.append(Minor(unused[-1] if unused else "", source=source(page.page_number, "Minor Programmes")))
                if not unused:
                    flag(minors[-1], "minor title not found")
                role = None
            elif not minors:
                continue
            role = parse_minor_table(table, minors[-1], role)
        titles_in_text += [clean(line).removeprefix("Minor in").strip()
                           for line in (page.extract_text() or "").splitlines() if clean(line).startswith("Minor in")]
    for minor in minors:
        if minor.description:
            exclusions = [clean(match.group(0)) for match in EXCLUSION.finditer(minor.description)]
            minor.exclusion_text = " ".join(exclusions) or None
        if minor.min_courses is None:
            flag(minor, "minimum courses/units not found")
        if not minor.core and not minor.electives:
            flag(minor, "no courses found")
    return minors


# ---------- (e) course descriptions ----------

def parse_prerequisites(text: str) -> list[list[str]] | None:
    """'CE F231 OR ME F212 and MATH F211' -> [['CE F231', 'ME F212'], ['MATH F211']]. None if no codes."""
    groups = [find_codes(part) for part in re.split(r"\band\b|&|;|,(?=\s*[A-Z]{2,5}\s+[A-Z]\d)", text, flags=re.I)]
    groups = [group for group in groups if group]
    return groups or None


def parse_descriptions(lines_by_page: list[tuple[int, list[str]]]) -> list[CourseDescription]:
    header = re.compile(r"^([A-Z]{2,5})\s+([A-Z])\s?(\d{3}[A-Z]?)\s+(.+?)" + UNITS_TAIL.pattern)
    stream = [(page, line) for page, lines in lines_by_page for line in join_split_codes(lines)]
    courses: list[CourseDescription] = []
    body: list[str] = []

    def finish() -> None:
        if not courses:
            return
        text = "\n".join(body)
        # why: words hyphenated across line breaks ("Fluid Me-\nchanics") are joined back.
        text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text).replace("\n", " ")
        split = re.split(r"\bPre[\s-]*requisites?\s*:?", text, maxsplit=1, flags=re.I)
        course = courses[-1]
        course.description = clean(split[0]) or None
        if len(split) > 1:
            course.prerequisites_text = clean(split[1])
            course.prerequisites = parse_prerequisites(split[1])
            if course.prerequisites is None:
                flag(course, "prerequisite text has no course codes")
        if course.description is None:
            flag(course, "no description text")

    for index, (page, line) in enumerate(stream):
        match = header.match(line)
        if match:
            finish()
            body = []
            numbers = match[5].split()
            as_int = [int(value.rstrip("*")) if value.rstrip("*").isdigit() else 0 for value in numbers]
            units = dict(zip("LPU", as_int)) if len(as_int) == 3 else {"L": None, "P": None, "U": as_int[0]}
            courses.append(CourseDescription(
                code=normalise_code(match[1], match[2], match[3]), title=clean(match[4]), **units,
                department=match[1], description=None, prerequisites=None, prerequisites_text=None,
                source=source(page, "Course Descriptions")))
            continue
        following = stream[index + 1][1] if index + 1 < len(stream) else ""
        # why: department headings ("Aeronautics") sit between courses; don't glue them into a description.
        if header.match(following) and len(line.split()) <= 6 and not re.search(r"[\d.,;:]", line):
            continue
        body.append(line)
    finish()
    share_paired_descriptions(courses)
    return courses


def share_paired_descriptions(courses: list[CourseDescription]) -> None:
    """'XXX F366 Laboratory Project' directly followed by 'XXX F367 ...' share the paragraph under the second.

    why: the Bulletin prints paired project courses as two header lines and one description.
    """
    for index in range(len(courses) - 2, -1, -1):
        course, following = courses[index], courses[index + 1]
        if course.description is None and following.description and course.note == "no description text":
            course.description = following.description
            course.prerequisites, course.prerequisites_text = following.prerequisites, following.prerequisites_text
            course.needs_verification, course.note = False, f"description shared with {following.code} (printed once for both)"


# ---------- orchestration ----------

def find_edition(page_texts: list[str]) -> str | None:
    """Most common 'academic year 2025-2026' in the text, as '2025-26'.

    why: the cover (with the edition) is an image; the fee pages state the academic year in text.
    """
    years = Counter(f"{start}-{end[-2:]}" for text in page_texts for start, end in
                    re.findall(r"academic year\s+(20\d\d)\s*-\s*(20\d\d|\d\d)", text, re.I))
    return years.most_common(1)[0][0] if years else None


def first_page_matching(page_texts: list[str], predicate, start: int = 1) -> int | None:
    return next((number for number in range(start, len(page_texts) + 1) if predicate(page_texts[number - 1])), None)


def attach_titles(programmes: list[Programme], known_titles: dict[str, str]) -> None:
    """Use the course-list / description title for named slots (chart titles are wrapped across lines)."""
    for programme in programmes:
        for chart in programme.semesters:
            for slot in chart.slots:
                if slot.code and slot.code in known_titles:
                    slot.title = known_titles[slot.code]


# Where the Bulletin contradicts itself, checked by hand against its own chart footers ("Discipline Core - 48 Units
# (16 Courses)"). Each fix: discipline -> (courses to remove, courses to add, reason).
CDC_CORRECTIONS = {
    "ENVIRONMENTAL AND SUSTAINABILITY ENGINEERING": (
        ["ENVS F232"],
        [("CHE F211", "Chemical Process Calculations", 3), ("BITS F240", "Introduction to Environmental & Sustainable Systems Engineering", 3),
         ("ENVS F323", "Sustainable Urban Design and Smart Cities", 3), ("ENVS F324", "Environmental Economics and Governance", 3)],
        "course list (p.322) prints 13 core courses; the chart (p.222) and its footer have 16 (48 units): "
        "adds CHE F211, BITS F240, ENVS F324 and numbers Sustainable Urban Design ENVS F323 (list: ENVS F232)"),
    "ELECTRONICS AND COMMUNICATION ENGINEERING": (
        ["ECE F331"], [("ECE F314", "Electromagnetic Fields & Microwave Engineering", 3)],
        "course list prints ECE F331 (4 units); every ECE chart uses ECE F314 (3 units), which gives the footer's 48 units"),
    "PHARMACY": (
        ["PHA F243"], [],
        "course list footnote: PHA F215 is offered in place of PHA F243 for students admitted 2014 onwards (footer: 16 courses)"),
}


def apply_cdc_corrections(course_lists: list[CourseList]) -> None:
    for course_list in course_lists:
        if course_list.discipline is None and any(course.code.startswith("BBA") for course in course_list.core):
            course_list.discipline = "BUSINESS ADMINISTRATION"  # the heading is an image in the PDF; the courses are BBA's
        fix = CDC_CORRECTIONS.get(course_list.discipline or "")
        if fix:
            removed, added, reason = fix
            course_list.core = [course for course in course_list.core if course.code not in removed]
            course_list.core += [ListedCourse(code, title, None, None, units) for code, title, units in added]
            course_list.note = f"{course_list.note}; corrected by hand: {reason}" if course_list.note else f"corrected by hand: {reason}"


def name_words(name: str) -> set[str]:
    """'B.E. Mathematic and Computing' / 'MATHEMATICS AND COMPUTING' -> {'mathematic', 'computing'} (for matching)."""
    name = re.sub(r"^(b\.\s?e\.|m\.\s?sc\.|b\.\s?pharm\.?|bachelor of)\s*", "", name.lower())
    words = re.findall(r"[a-z]+", name.replace("&", " and "))
    return {w.rstrip("s") for w in words if w not in {"and", "engineering", "in", "with", "specialization", "of", "the", "honour", "honours", "b", "e"}}


def best_list(name: str, course_lists: list[CourseList], charted: set[str]) -> CourseList | None:
    """The course list whose discipline name matches best; ties broken by shared chart courses."""
    wanted = name_words(name)
    scored = [(len(wanted & name_words(cl.discipline)) / len(wanted | name_words(cl.discipline)),
               len(charted & {c.code for c in cl.core}), cl) for cl in course_lists if cl.discipline]
    score, overlap, found = max(scored, key=lambda item: (item[0], item[1]))
    return found if score >= 0.5 or overlap >= 5 else None


def link_cdc_lists(programmes: list[Programme], course_lists: list[CourseList]) -> None:
    """programme.cdc_lists: the course list(s) whose core courses are this programme's CDCs
    (dual degrees: one per component)."""
    for programme in programmes:
        if programme.type == "single":
            charted = {slot.code for chart in programme.semesters for slot in chart.slots if slot.code}
            found = best_list(programme.name, course_lists, charted)
            programme.cdc_lists = [found.discipline] if found else []
        elif programme.type == "dual":
            programme.cdc_lists = [found.discipline for name in programme.components if (found := best_list(name, course_lists, set()))]
        if programme.type != "dual_template" and len(programme.cdc_lists) < max(1, len(programme.components)):
            flag(programme, "a course list for this programme (or one of its components) was not found")


def extract_all(pdf_path: Path) -> dict:
    with pdfplumber.open(pdf_path) as pdf:
        log.info("Reading %d pages to find the sections...", len(pdf.pages))
        page_texts = [page.extract_text() or "" for page in pdf.pages]
        edition = find_edition(page_texts)

        pattern_start = first_page_matching(page_texts, lambda text: bool(PATTERN_TITLE.match(text.strip())))
        lists_start = first_page_matching(page_texts, lambda text: COURSE_LISTS_START in text, pattern_start or 1)
        minors_start = first_page_matching(page_texts, lambda text: bool(MINORS_START.search(text)), lists_start or 1)
        descriptions_start = first_page_matching(page_texts, lambda text: DESCRIPTIONS_START in text, minors_start or 1)
        descriptions_end = first_page_matching(page_texts, lambda text: bool(DESCRIPTIONS_END.search(text)), descriptions_start or 1)
        if None in (pattern_start, lists_start, minors_start, descriptions_start, descriptions_end):
            raise SystemExit("Could not find every Bulletin section heading; has the format changed?")
        # why: minors run until the next section; the first page after them with no 'Minor in' table ends it.
        minors_end = first_page_matching(
            page_texts, lambda text: "Minor in" not in text and not CODE.search(text[:300]), minors_start + 1) or descriptions_start

        log.info("Programme charts: pages %d-%d", pattern_start, lists_start - 1)
        programmes = extract_programmes(pdf, pattern_start, lists_start - 1, edition)
        log.info("Course lists: pages %d-%d", lists_start, minors_start - 1)
        list_lines = [(number, column_lines(pdf.pages[number - 1])) for number in range(lists_start, minors_start)]
        course_lists, huel_pool, audit, project_patterns = parse_course_lists(list_lines)
        log.info("Minors: pages %d-%d", minors_start, minors_end - 1)
        minors = extract_minors(pdf, minors_start, minors_end - 1)
        log.info("Course descriptions: pages %d-%d", descriptions_start, descriptions_end - 1)
        description_lines = [(number, column_lines(pdf.pages[number - 1])) for number in range(descriptions_start, descriptions_end)]
        descriptions = parse_descriptions(description_lines)

    apply_cdc_corrections(course_lists)
    link_cdc_lists(programmes, course_lists)
    known_titles = {course.code: course.title for course in descriptions}
    known_titles.update({course.code: course.title for course_list in course_lists for course in course_list.core + course_list.discipline_electives})
    attach_titles(programmes, known_titles)

    return {
        "edition": edition,
        "source": {"doc": pdf_path.name},
        "edition_note": "read from the 'academic year' stated in the fee pages; the cover is an image",
        "programmes": programmes,
        "course_lists": course_lists,
        "huel_pool": huel_pool,
        "huel_pool_source": source(lists_start, "Pool of Humanities courses for first degree programmes"),
        "audit_courses": audit,
        "project_course_patterns": project_patterns,
        "minors": minors,
        "course_descriptions": descriptions,
    }


def to_json(value):
    """dataclasses (nested in dicts/lists) -> plain JSON values."""
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return {key: to_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json(item) for item in value]
    return value


if __name__ == "__main__":
    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH}")
    bulletin = extract_all(PDF_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(to_json(bulletin), indent=2, ensure_ascii=False))
    for key in ("programmes", "course_lists", "minors", "course_descriptions"):
        records = bulletin[key]
        log.info("%s: %d records (%d flagged)", key, len(records), sum(record.needs_verification for record in records))
    log.info("huel_pool: %d, audit_courses: %d. Saved to %s", len(bulletin["huel_pool"]), len(bulletin["audit_courses"]), OUTPUT_PATH)
