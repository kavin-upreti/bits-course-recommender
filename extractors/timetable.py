import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# why: build paths from this file's location, not the terminal's cwd,
# so the script works no matter where you run it from or where the project folder lives.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "dataset" / "raw" / "timetable.pdf"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "code processed" / "timetable.json"

# A table whose first row contains this text has a header row.
HEADER_MARKER = "COURSE NO"

# Timetable fields -> regex matched against each column's header text.
# why: columns are found by name, so a table with reordered or extra columns still parses.
TIMETABLE_COLUMNS = {
    "com_code": r"COM COD",
    "course_no": r"COURSE NO",
    "title": r"TITLE",
    "lecture": r"^L$",
    "practical": r"^P$",
    "tutorial": r"^T$",
    "self_study": r"^S$",  # legend: S = self-study hours per week
    "units": r"^U",
    "section": r"^SEC",
    "instructor": r"INSTRUCTOR",
    "room": r"ROOM",
    "timings": r"DAYS",
    "midsem": r"MIDSEM",
    "compre": r"COMPRE",
}
EQUIVALENT_COLUMN = r"EQUIVALENT"

SECTION_TYPES = {"L": "lecture", "T": "tutorial", "P": "practical"}
DAYS = {"M", "T", "W", "Th", "F", "S", "Su"}
CANCELLED = "CANCLED"  # sic: this is how the PDF spells it
# Footnote on every timetable page: "Courses with com cod >=5000 are meant only for
# 2026 admissions into FD, HD and PHD and not for others".
RESTRICTED_COM_CODE = 5000

Row = list[str | None]  # one raw table row from pdfplumber; None = cell covered by a merged cell
Record = dict[str, str]  # one timetable row keyed by field name, e.g. {"course_no": "BIO F101", ...}


@dataclass
class Section:
    instructors: list[str]
    room: str | None
    timings: dict[str, list[int]]  # day -> hour numbers, e.g. {"M": [3], "W": [3], "Th": [9]}
    cancelled: bool = False  # room printed as "CANCLED"; kept so the data shows it existed


@dataclass
class Course:
    course_no: str
    title: str
    com_code: int | None
    allowed_for_before_2026_batch: bool  # False = only for 2026 FD/HD/PhD admissions; earlier batches can't take it
    credits: dict[str, int]
    midsem: str | None = None  # raw, e.g. "09/10 AN1"
    midsem_date: str | None = None  # "09/10" (dd/mm)
    midsem_session: str | None = None  # "AN1"; clock times are in manually processed/timetable.json
    compre: str | None = None
    compre_date: str | None = None
    compre_session: str | None = None
    instructor_in_charge: str | None = None  # the all-caps name in the timetable; None if none printed
    sections: dict[str, dict[str, Section]] = field(default_factory=dict)
    equivalents: list[str] = field(default_factory=list)
    source_pages: list[int] = field(default_factory=list)
    needs_verification: bool = False
    note: str | None = None  # why it was flagged


@dataclass
class Timetable:
    semester: str | None  # as printed, e.g. "FIRST SEMESTER 2026-27"
    semester_tag: str | None  # "2026-27 Sem 1"
    campus: str | None  # "Pilani"
    source: dict[str, str]
    courses: list[Course]


@dataclass
class Layout:
    """Column layout of the last header seen, reused by headerless continuation tables."""

    kind: str  # "timetable" or "equivalents"
    columns: dict[str, int]
    width: int
    equivalent_columns: list[int] = field(default_factory=list)


def clean(cell: str | None) -> str:
    """Strip a cell and collapse line breaks from wrapped text into single spaces."""
    return " ".join((cell or "").split())


def to_int(text: str) -> int:
    """Credit cells hold a number or '-' (meaning 0)."""
    return int(text) if text.isdigit() else 0


def to_com_code(text: str) -> int | None:
    return int(text) if text.isdigit() else None


def is_instructor_in_charge(name: str) -> bool:
    """The timetable prints the instructor-in-charge in ALL CAPS, other instructors in mixed case."""
    return name == name.upper() and any(char.isalpha() for char in name)


def parse_timings(text: str, problems: list[str]) -> dict[str, list[int]]:
    """Parse 'M W 3 Th 9' into {'M': [3], 'W': [3], 'Th': [9]}.

    why: the format is groups of <days...> <hours...>; every day in a group meets at
    every hour in that group. 'M 6 7' = Monday hours 6 and 7 (a 2-hour lab).
    Hours are the timetable's period numbers, not clock times.
    """
    schedule: dict[str, list[int]] = {}
    group_days: list[str] = []
    previous_was_hour = False

    for token in text.split():
        if token.isdigit():
            if not group_days:
                problems.append(f"timing {text!r}: hour {token} has no day before it")
            for day in group_days:
                schedule.setdefault(day, []).append(int(token))
            previous_was_hour = True
        elif token in DAYS:
            if previous_was_hour:  # a day after hours starts a new group
                group_days = []
            group_days.append(token)
            previous_was_hour = False
        else:
            problems.append(f"timing {text!r}: unknown token {token!r}")

    return schedule


def flag(course: Course, reason: str) -> None:
    course.needs_verification = True
    course.note = f"{course.note}; {reason}" if course.note else reason


def split_exam(text: str | None) -> tuple[str | None, str | None]:
    """'09/10 AN1' -> ('09/10', 'AN1'); '14/12 FN' -> ('14/12', 'FN'). Anything else -> (None, None)."""
    match = re.fullmatch(r"(\d{1,2}/\d{1,2})\s*([A-Z]{2}\d?)", text or "")
    return (match[1], match[2]) if match else (None, None)


def section_type(section_id: str) -> str | None:
    """Map a section ID to lecture/tutorial/practical.

    why: most IDs are simple (L1, T2, P10), but some carry extra parts: 'L1T1' is a
    tutorial attached to lecture L1, and 'L1AJ' / 'T2RM' have a suffix. The last
    letter+number pair is the one that decides the type.
    """
    pairs = re.findall(r"([LTP])\d+", section_id)
    return SECTION_TYPES[pairs[-1]] if pairs else None


def read_header(table: list[Row]) -> tuple[list[str], int]:
    """Return (column names, number of header rows) for a table that starts with a header.

    why: a None in the first row means a merged cell spanning several columns
    (e.g. CREDIT over L / P / T / S / U). The real names of those columns are in
    the second row, so the two rows are combined.
    """
    top = table[0]
    if None in top and len(table) > 1:
        names = [clean(sub) or clean(main) for main, sub in zip(top, table[1])]
        return names, 2
    return [clean(cell) for cell in top], 1


def find_columns(names: list[str], patterns: dict[str, str]) -> dict[str, int] | None:
    """Map each field to the index of the first column whose name matches its pattern."""
    columns: dict[str, int] = {}
    for field_name, pattern in patterns.items():
        index = next((i for i, name in enumerate(names) if re.search(pattern, name)), None)
        if index is None:
            return None
        columns[field_name] = index
    return columns


def detect_layout(names: list[str]) -> Layout | None:
    """Work out which kind of table a header belongs to, or None if we don't need it."""
    equivalent_columns = [i for i, name in enumerate(names) if re.search(EQUIVALENT_COLUMN, name)]
    if equivalent_columns:
        columns = find_columns(names, {"course_no": TIMETABLE_COLUMNS["course_no"]})
        if columns:
            return Layout("equivalents", columns, len(names), equivalent_columns)

    columns = find_columns(names, TIMETABLE_COLUMNS)
    if columns:
        return Layout("timetable", columns, len(names))
    return None


def to_record(row: Row, columns: dict[str, int]) -> Record:
    return {field_name: clean(row[index]) for field_name, index in columns.items()}


def parse_timetable(records: list[tuple[int, Record]]) -> list[Course]:
    """Turn the timetable rows of the whole PDF (in page order) into courses.

    Blank cells mean "same as above": a row with no course number belongs to the
    previous course, and a row with no section adds another instructor to the
    previous section. Because all pages are parsed as one stream, a course that
    continues onto the next page is handled naturally.
    """
    courses: list[Course] = []
    course: Course | None = None
    section: Section | None = None

    for page_number, record in records:
        if record["course_no"]:
            com_code = to_com_code(record["com_code"])
            course = Course(
                course_no=record["course_no"],
                title=record["title"],
                com_code=com_code,
                allowed_for_before_2026_batch=com_code is None or com_code < RESTRICTED_COM_CODE,
                credits={key: to_int(record[key]) for key in ("lecture", "practical", "tutorial", "self_study", "units")},
            )
            courses.append(course)
            section = None

        if course is None:
            log.warning("Page %d: row before any course, skipped: %s", page_number, record)
            continue

        if page_number not in course.source_pages:
            course.source_pages.append(page_number)

        section_id, instructor = record["section"], record["instructor"]

        if is_instructor_in_charge(instructor):
            # why: ALL CAPS only marks the IC; store it like the other names ("Jitendra Panwar").
            instructor = instructor.title()
            if course.instructor_in_charge is None:
                course.instructor_in_charge = instructor

        if not section_id:
            if section and instructor:
                section.instructors.append(instructor)
            continue

        # Exam dates are printed on section rows, not the course row; keep the first seen.
        course.midsem = course.midsem or record["midsem"] or None
        course.compre = course.compre or record["compre"] or None

        kind = section_type(section_id)
        if kind is None:
            flag(course, f"unknown section type {section_id!r} on page {page_number}")
            section = None  # why: so instructor-only rows below it aren't attached to the previous section
            continue

        problems: list[str] = []
        cancelled = record["room"] == CANCELLED
        section = Section(
            instructors=[instructor] if instructor else [],
            room=None if cancelled else record["room"] or None,
            timings={} if cancelled else parse_timings(record["timings"], problems),
            cancelled=cancelled,
        )
        for problem in problems:
            flag(course, f"{section_id}: {problem}")
        course.sections.setdefault(kind, {})[section_id] = section

    return courses


def extract_all(pdf_path: Path) -> list[Course]:
    """Read every table in the PDF, route it by its header, and build the course list."""
    timetable_records: list[tuple[int, Record]] = []
    equivalents: dict[str, list[str]] = {}
    layout: Layout | None = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table:
                    continue

                if any(HEADER_MARKER in clean(cell) for cell in table[0]):
                    names, header_rows = read_header(table)
                    layout = detect_layout(names)
                    body = table[header_rows:]
                elif layout and len(table[0]) == layout.width:
                    # why: some pages hold a few leftover rows with no header; they
                    # continue the last table, so reuse its column layout.
                    body = table
                else:
                    continue

                if layout is None:
                    continue
                for row in body:
                    if layout.kind == "timetable":
                        timetable_records.append((page.page_number, to_record(row, layout.columns)))
                    else:
                        course_no = clean(row[layout.columns["course_no"]])
                        others = [clean(row[i]) for i in layout.equivalent_columns]
                        equivalents.setdefault(course_no, []).extend(
                            code for code in others if code and code != course_no
                        )

    courses = parse_timetable(timetable_records)
    for course in courses:
        course.equivalents = equivalents.get(course.course_no, [])
        course.midsem_date, course.midsem_session = split_exam(course.midsem)
        course.compre_date, course.compre_session = split_exam(course.compre)
        if course.midsem and course.midsem_date is None:
            flag(course, f"midsem {course.midsem!r} not in 'dd/mm SESSION' form")
        if course.compre and course.compre_date is None:
            flag(course, f"compre {course.compre!r} not in 'dd/mm SESSION' form")
        if course.com_code is None:
            flag(course, "no COM COD printed")
    return courses


def read_semester_and_campus(pdf_path: Path) -> tuple[str | None, str | None, str | None]:
    """(printed semester, semester tag, campus) from the cover and calendar pages.

    why: read from the PDF, not hardcoded, so next semester's timetable tags itself.
    """
    with pdfplumber.open(pdf_path) as pdf:
        front = "\n".join(page.extract_text() or "" for page in pdf.pages[:3])
    semester = re.search(r"(FIRST|SECOND)\s+SEMESTER\s+(\d{4})\s*-\s*(\d{2,4})", front, re.I)
    campus = re.search(r"([A-Z][A-Za-z]+)\s+CAMPUS", front)
    if semester is None:
        log.warning("Semester not found on the first pages")
        return None, None, campus[1].title() if campus else None
    number = 1 if semester[1].upper() == "FIRST" else 2
    tag = f"{semester[2]}-{semester[3][-2:]} Sem {number}"
    return " ".join(semester[0].upper().split()), tag, campus[1].title() if campus else None


def save(timetable: Timetable, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(timetable), indent=2))


def normalise_code(text: str) -> str:
    """'cs f111', 'CSF111' and 'CS  F111' all become 'CSF111'."""
    return re.sub(r"\s+", "", text).upper()


def find_courses(courses: list[Course], query: str) -> list[Course]:
    """Match by course number first; if nothing matches, search titles."""
    code = normalise_code(query)
    by_code = [course for course in courses if normalise_code(course.course_no) == code]
    if by_code:
        return by_code
    words = query.upper().split()
    return [course for course in courses if all(word in course.title for word in words)]


def ask_loop(courses: list[Course]) -> None:
    """Keep asking for a course until the user enters nothing."""
    while True:
        query = input("\nCourse number or title (Enter to quit): ").strip()
        if not query:
            return

        matches = find_courses(courses, query)
        # why: a title search can match several courses; list them instead of guessing.
        # A course-number match can also return 2 entries (same course, different COM COD).
        if not matches:
            print("No course found.")
        elif len({course.course_no for course in matches}) > 1:
            print("Several courses match, type the course number:")
            for course in matches:
                print(f"  {course.course_no:<12} {course.title}")
        else:
            for course in matches:
                print(json.dumps(asdict(course), indent=2))


if __name__ == "__main__":
    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH}")

    printed_semester, semester_tag, campus = read_semester_and_campus(PDF_PATH)
    all_courses = extract_all(PDF_PATH)
    save(Timetable(printed_semester, semester_tag, campus, {"doc": PDF_PATH.name}, all_courses), OUTPUT_PATH)
    flagged = sum(course.needs_verification for course in all_courses)
    log.info("Saved %d course entries (%d flagged for verification) to %s", len(all_courses), flagged, OUTPUT_PATH)
    if sys.stdin.isatty():
        ask_loop(all_courses)
