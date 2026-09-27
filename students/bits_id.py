"""Parse a BITS ID into batch, branch code(s) and campus, and resolve the codes to catalog Programmes.

Formats (from the brief):  2025A7PS0832P  (single degree A7)
                           2025B3A7PS0832P (dual: M.Sc. B3 + B.E. A7; the B-code always comes first)
"""
import re
from dataclasses import dataclass

from catalog.models import Programme

# Branch code -> exact Programme.name in the Bulletin data. Only the codes given in the brief's table;
# programmes without a code (Robotics, Biotech, BBA, ...) can't be reached from an ID, on purpose.
BRANCH_CODES: dict[str, str] = {
    "A1": "B.E. Chemical",
    "A2": "B.E. Civil",
    "A3": "B.E. Electrical & Electronics",
    "A4": "B.E. Mechanical",
    "A5": "B. Pharm.",
    "A7": "B.E. Computer Science",
    "A8": "B.E. Electronics and Instrumentation",
    "AA": "B.E. Electronics & Communication",
    "AB": "B.E. Manufacturing",
    "AC": "B.E. Electronics & Computer Engineering",
    "AD": "B.E. Mathematics and Computing",
    "AJ": "B.E. Environmental and Sustainability Engineering",
    "B1": "M.Sc. Biological Sciences",
    "B2": "M.Sc. Chemistry",
    "B3": "M.Sc. Economics",
    "B4": "M.Sc. Mathematics",
    "B5": "M.Sc. Physics",
    "B7": "M.Sc. Semiconductor and Nanoscience",
}
CAMPUSES: dict[str, str] = {"P": "Pilani", "G": "Goa", "H": "Hyderabad", "D": "Dubai"}

ID_PATTERN = re.compile(r"^(?P<year>\d{4})(?P<first>[AB][0-9A-Z])(?P<second>A[0-9A-Z])?(?:PS|TS)\d{4}(?P<campus>[A-Z])$")


class BitsIdError(ValueError):
    """The ID doesn't match the format, or names a branch / combination we have no data for."""


@dataclass(frozen=True)
class ParsedId:
    admission_year: int
    first_code: str
    second_code: str | None
    campus: str


def parse_bits_id(raw_id: str) -> ParsedId:
    """Split an ID into its parts. Pure string work; doesn't touch the database."""
    bits_id = raw_id.strip().upper()
    match = ID_PATTERN.match(bits_id)
    if not match:
        raise BitsIdError("Expected an ID like 2025A7PS0832P or 2025B3A7PS0832P.")
    first, second = match["first"], match["second"]
    if second and not first.startswith("B"):
        raise BitsIdError("A dual-degree ID starts with the M.Sc. code (B...), then the B.E. code (A...).")
    for code in filter(None, (first, second)):
        if code not in BRANCH_CODES:
            raise BitsIdError(f"Unknown branch code {code}.")
    if match["campus"] not in CAMPUSES:
        raise BitsIdError(f"Unknown campus letter {match['campus']}.")
    return ParsedId(int(match["year"]), first, second, CAMPUSES[match["campus"]])


def resolve_programme(parsed: ParsedId) -> Programme:
    """The student's Programme. For a dual ID this is the combined dual chart, which links both degrees
    through first_component / second_component.

    why the dual chart and not two singles: the Bulletin gives every M.Sc. + B.E. pair its own chart and lists.
    """
    first = Programme.objects.filter(name=BRANCH_CODES[parsed.first_code], type="single").first()
    if first is None:
        raise BitsIdError(f"No Bulletin data for {BRANCH_CODES[parsed.first_code]}.")
    if not parsed.second_code:
        return first
    dual = Programme.objects.filter(
        type="dual", first_component=first, second_component__name=BRANCH_CODES[parsed.second_code]
    ).first()
    if dual is None:
        raise BitsIdError(f"The Bulletin has no dual-degree chart for {first.name} + {BRANCH_CODES[parsed.second_code]}.")
    return dual


def second_degree_options(first_code: str) -> list[tuple[str, str]]:
    """(code, name) of every B.E. an M.Sc. can pair with: the ones the Bulletin gives a dual chart for."""
    names = set(Programme.objects.filter(type="dual", first_component__name=BRANCH_CODES.get(first_code))
                .values_list("second_component__name", flat=True))
    return [(code, name) for code, name in BRANCH_CODES.items() if code.startswith("A") and name in names]
