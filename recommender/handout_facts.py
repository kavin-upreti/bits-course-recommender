"""Handout rows -> one set of clean, tri-state facts per course.

Every handout filter, dislike penalty and card reads handouts only through `get_course_facts`. A value the handouts
don't state is None ("couldn't verify"), never False. A course with several handouts gets a value only when they agree.
"""
import re
from collections import Counter
from dataclasses import dataclass, field

from catalog.models import Course, Handout

# the extractor's component kinds (checked on the real data)
MIDSEM, COMPRE, QUIZ, PROJECT = "midsem", "compre", "quiz", "project"
OPEN_BOOK = "OB"
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
NUMBER = r"(\d+|" + "|".join(NUMBER_WORDS) + r")"
QUIZ_WORD = r"(?:quiz\w*|quizes|tests?)"
# "Best 5 out of 6", "(3 out of 4)", "Best 2 of 3" -> the total held
BEST_OF = re.compile(r"\b(\d+)\s+(?:out\s+)?of\s+(\d+)\b", re.I)
RANGE = re.compile(r"\b(\d+|[IVX]+)\s*(?:-|–|—|to)\s*(\d+|[IVX]+)\b")
LIST = re.compile(r"\b\d+(?:\s*(?:,|&|and)\s*\d+)+\b", re.I)
# "3 Quizzes", "3 Class tests", "(4 quizzes during ...)", "Quizzes (3)", "Quiz (One)", "Quizzes(two;..."
COUNT_BEFORE = re.compile(NUMBER + r"\s+(?:\w+\s+)?" + QUIZ_WORD + r"\b", re.I)
COUNT_AFTER = re.compile(QUIZ_WORD + r"\s*\(\s*" + NUMBER + r"\b", re.I)
# Makeup text the extractor's NLI couldn't decide (it only asks "can be given" / "never allowed"): read at fact time.
NO_MAKEUP_AT_ALL = re.compile(r"\bno\s+make\s*-?\s*ups?\s+(?:\w+\s+){0,3}(?:for|in)\s+this\s+course", re.I)
CONDITIONAL_MAKEUP = re.compile(r"only|genuine|emergenc|illness|medical|hospitali|permission|discretion|valid (?:reason|proof)"
                                r"|certificate|will be (?:given|granted|considered|allowed)|to be granted|consider", re.I)
DEFAULT_POLICY = re.compile(r"as per (?:the )?(?:institute|augs|agsr|academic)|institute(?:'s|’s)? (?:rules|policy|guidelines)"
                            r"|see part[- ]i\b", re.I)
SENTENCE = re.compile(r"^(.+?[.!?])(?:\s|$)")
PLURAL = re.compile(r"quizzes|quizes|quiz\s*\(\s*(?:zes|s)\s*\)|\btests\b", re.I)

# fields that are None when the handouts don't say; `completeness` counts the known ones
TRI_STATE = ("has_midsem", "quiz_count", "compre_percent", "project_percent", "open_book",
             "attendance_required", "makeup_lenient", "makeup_allowed")
# midsem_percent / quiz_percent are for display only (cards: "Yes (30%)"), so they aren't counted in completeness
COMBINED = TRI_STATE + ("midsem_percent", "quiz_percent")
EVALUATION_FIELDS = ("has_midsem", "midsem_percent", "quiz_count", "quiz_percent", "compre_percent", "project_percent",
                     "open_book")


@dataclass
class CourseFacts:
    code: str
    evaluation_parsed: bool        # True if at least one handout of this course has a non-empty evaluation list
    has_midsem: bool | None = None
    midsem_percent: float | None = None
    quiz_count: int | None = None
    quiz_percent: float | None = None   # weight of components whose name says "quiz"
    compre_percent: float | None = None
    project_percent: float | None = None
    open_book: bool | None = None
    attendance_required: bool | None = None
    attendance_follows_default: bool = False   # handout says "as per AUGS/AGSR guidelines"
    makeup_lenient: bool | None = None
    makeup_allowed: bool | None = None
    makeup_condition: str = ""                 # the handout's words when makeups are only granted on conditions
    makeup_follows_default: bool = False       # "as per AUGSD guidelines"
    notes: dict[str, str] = field(default_factory=dict)  # field name -> short note, e.g. {"quiz_count": "handouts differ"}
    sources: list[str] = field(default_factory=list)     # handout file names used

    @property
    def completeness(self) -> int:
        """How many of the tri-state facts are known (0-8)."""
        return sum(getattr(self, name) is not None for name in TRI_STATE)

    @property
    def has_handout(self) -> bool:
        return bool(self.sources)

    def note_for(self, name: str) -> str:
        """Why a fact is unknown, in a few words; "" when the handout simply doesn't say."""
        if not self.has_handout:
            return "no handout"
        if name in self.notes:
            return self.notes[name]
        if name in EVALUATION_FIELDS and not self.evaluation_parsed:
            return "handout doesn't list its evaluation"
        if name == "attendance_required" and self.attendance_follows_default:
            return "follows institute rules"
        if name in ("makeup_allowed", "makeup_lenient") and self.makeup_follows_default:
            return "follows institute rules"
        return ""

    def unknown_text(self, name: str) -> str:
        """"couldn't verify", with the reason in brackets when there is one."""
        note = self.note_for(name)
        return f"couldn't verify ({note})" if note else "couldn't verify"


_cache: dict[str, CourseFacts] = {}


def clear_facts_cache() -> None:
    """Forget every computed CourseFacts (the catalog was reloaded)."""
    _cache.clear()


def _to_int(token: str) -> int:
    """"3" / "three" / "III" -> 3."""
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token.lower()) or ROMAN[token.upper()]


def quiz_component_count(name: str) -> int | None:
    """How many quizzes one evaluation component stands for; None when the name is plural without a number."""
    if match := BEST_OF.search(name):
        return int(match[2])
    if match := RANGE.search(name):
        first, last = _to_int(match[1]), _to_int(match[2])
        if last >= first:
            return last - first + 1
    if match := LIST.search(name):
        return len(re.findall(r"\d+", match[0]))
    if match := COUNT_BEFORE.search(name) or COUNT_AFTER.search(name):
        return _to_int(next(group for group in match.groups() if group))
    return None if PLURAL.search(name) else 1


def _quiz_count(evaluation: list[dict]) -> tuple[int | None, str]:
    """Sum over the quiz components, and a note when the count isn't stated."""
    total = 0
    for component in evaluation:
        if component["kind"] != QUIZ:
            continue
        count = quiz_component_count(component["name"])
        if count is None:
            return None, "quiz count not stated"
        total += count
    return total, ""


def _weight(evaluation: list[dict], kind: str) -> float:
    return round(sum(component["weightage_percent"] or 0 for component in evaluation if component["kind"] == kind), 2)


def _makeup_lenient(handout: Handout, allowed: bool | None) -> bool | None:
    """False if makeups aren't allowed or the midsem / compre has none; True if allowed and neither is refused."""
    per_component = handout.makeup_per_component or {}
    exams = [per_component.get(MIDSEM), per_component.get(COMPRE)]
    if allowed is False or False in exams:
        return False
    return True if allowed is True else None


def makeup_from_text(text: str) -> tuple[bool | None, str, bool]:
    """(allowed, condition, follows_default) read from the makeup text, for when the extractor left allowed unknown.
    "To be granted only in case of serious illness" -> (True, that sentence, False): makeups exist, on a condition.
    why a rule here and not in the extractor: its NLI hypotheses have no "only if" answer; 45 of 50 such texts fit these."""
    text = " ".join((text or "").split())
    if not text:
        return None, "", False
    if NO_MAKEUP_AT_ALL.search(text):
        return False, "", False
    if DEFAULT_POLICY.search(text):
        return None, "", True
    if CONDITIONAL_MAKEUP.search(text):
        first = SENTENCE.match(text)
        return True, (first.group(1) if first else text)[:160], False
    return None, "", False


def handout_facts(handout: Handout) -> tuple[dict, dict[str, str]]:
    """The facts of one handout, plus notes for the ones it can't give."""
    evaluation = handout.evaluation or []
    parsed = bool(evaluation)
    notes: dict[str, str] = {}
    facts: dict = {"evaluation_parsed": parsed}
    if parsed:
        facts["has_midsem"] = any(component["kind"] == MIDSEM for component in evaluation)
        facts["midsem_percent"] = _weight(evaluation, MIDSEM)
        facts["quiz_count"], quiz_note = _quiz_count(evaluation)
        if quiz_note:
            notes["quiz_count"] = quiz_note
        # why the name and not the kind: the student asks "are there quizzes?", and the name answers that directly
        facts["quiz_percent"] = round(sum(component["weightage_percent"] or 0 for component in evaluation
                                          if "quiz" in component["name"].lower()), 2)
        facts["compre_percent"] = _weight(evaluation, COMPRE)
        facts["project_percent"] = _weight(evaluation, PROJECT)
        facts["open_book"] = any(component.get("nature") == OPEN_BOOK for component in evaluation) \
            or (handout.open_book_percent or 0) > 0
    else:
        facts.update(dict.fromkeys(EVALUATION_FIELDS))
    facts["attendance_required"] = handout.attendance_required
    facts["attendance_follows_default"] = handout.attendance_follows_default
    allowed, condition, follows_default = (handout.makeup_allowed, "", handout.makeup_follows_default)
    if allowed is None:
        allowed, condition, follows_default = makeup_from_text(handout.makeup_text)
        follows_default = follows_default or handout.makeup_follows_default
    facts["makeup_allowed"] = allowed
    facts["makeup_lenient"] = _makeup_lenient(handout, allowed)
    facts["makeup_condition"] = condition
    facts["makeup_follows_default"] = follows_default
    return facts, notes


def _combine(values: list, notes: list[str]) -> tuple[object, str]:
    """All known values equal -> that value; they differ -> None, "handouts differ"; none known -> None."""
    known = [value for value in values if value is not None]
    if known and all(value == known[0] for value in known):
        return known[0], ""
    if known:
        return None, "handouts differ"
    return None, next((note for note in notes if note), "")


def compute_course_facts(code: str, handouts: list[Handout]) -> CourseFacts:
    """Combine the facts of a course's handouts."""
    if not handouts:
        return CourseFacts(code=code, evaluation_parsed=False, notes={"all": "no handout"})
    per_handout = [handout_facts(handout) for handout in sorted(handouts, key=lambda handout: handout.file)]
    result = CourseFacts(
        code=code, evaluation_parsed=any(facts["evaluation_parsed"] for facts, _ in per_handout),
        attendance_follows_default=any(facts["attendance_follows_default"] for facts, _ in per_handout),
        makeup_follows_default=any(facts["makeup_follows_default"] for facts, _ in per_handout),
        makeup_condition=next((facts["makeup_condition"] for facts, _ in per_handout if facts["makeup_condition"]), ""),
        sources=sorted(handout.file for handout in handouts),
    )
    for name in COMBINED:
        value, note = _combine([facts[name] for facts, _ in per_handout], [notes.get(name, "") for _, notes in per_handout])
        setattr(result, name, value)
        if note:
            result.notes[name] = note
    return result


def get_course_facts(course: Course) -> CourseFacts:
    """Cached per course code; cleared by `ingest`."""
    if course.code not in _cache:
        _cache[course.code] = compute_course_facts(course.code, list(Handout.objects.filter(course=course)))
    return _cache[course.code]


def all_course_facts() -> dict[str, CourseFacts]:
    """Facts for every course, in two queries (fills the cache)."""
    by_course: dict[str, list[Handout]] = {}
    for handout in Handout.objects.select_related("course"):
        by_course.setdefault(handout.course.code, []).append(handout)
    for code in Course.objects.values_list("code", flat=True):
        if code not in _cache:
            _cache[code] = compute_course_facts(code, by_course.get(code, []))
    return dict(_cache)


def number(value: float) -> int | float:
    """40.0 -> 40, 12.5 -> 12.5."""
    return int(value) if float(value).is_integer() else value


def attendance_text(facts: CourseFacts) -> str | None:
    """"required" / "not required" / "follows institute rules", or None when unknown."""
    if facts.attendance_required is not None:
        return "required" if facts.attendance_required else "not required"
    return "follows institute rules" if facts.attendance_follows_default else None


def makeup_text(facts: CourseFacts) -> str | None:
    """What the handouts say about makeups, or None when unknown."""
    if facts.makeup_allowed is False:
        return "not allowed"
    if facts.makeup_condition:
        return f"only on conditions: “{facts.makeup_condition}”"
    if facts.makeup_lenient:
        return "allowed for midsem and compre"
    if facts.makeup_lenient is False:
        return "not allowed for the midsem or compre"
    if facts.makeup_allowed:
        return "allowed"
    return "follows institute rules" if facts.makeup_follows_default else None


def facts_report() -> str:
    """Unknowns per fact, evaluation kinds and a sample of quiz names with their counts (for eyeballing the rules)."""
    facts = [facts for facts in all_course_facts().values() if facts.has_handout]
    lines = [f"Courses with at least one handout: {len(facts)}", "", "Unknown (None) per fact:"]
    for name in TRI_STATE:
        lines.append(f"  {name:22} {sum(getattr(item, name) is None for item in facts):4}")
    kinds: Counter = Counter()
    quiz_names: set[str] = set()
    for handout in Handout.objects.all():
        for component in handout.evaluation or []:
            kinds[component["kind"]] += 1
            if component["kind"] == QUIZ:
                quiz_names.add(component["name"])
    lines += ["", "Evaluation component kinds:"] + [f"  {kind:22} {count:4}" for kind, count in kinds.most_common()]
    lines += ["", "Quiz component names -> count (first 20 alphabetically):"]
    lines += [f"  {quiz_component_count(name)!s:>4}  {name}" for name in sorted(quiz_names)[:20]]
    return "\n".join(lines)
