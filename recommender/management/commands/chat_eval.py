"""`python manage.py chat_eval`: the whole assistant, real LLM, on recommender/eda/chat_cases.json.

Each case has a student, a message and the IDEAL get_eligible_courses call(s) a person would make for it. The
pipeline's answer to the ideal calls is the reference, so the two error sources are measured apart:
- call match: did the LLM call get_eligible_courses with the ideal category / filters / count / preferences?
  (topics aren't compared word for word: the next number says whether its topics found the same courses)
- reference recall: share of the reference courses that reach the student's cards.
- invented codes: the guardrail had to act. Ranking quality itself is measured without any LLM by embedding_eda.
Writes eda/chat_eval.md. Uses the free LLM quota: ~2-4 calls per case.
"""
import json
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from recommender.agent import run_agent
from recommender.context import build_context
from recommender.eligible import run
from recommender.validation import validate_call
from students.models import Student

CASES = Path(__file__).resolve().parents[2] / "eda" / "chat_cases.json"
OUT = settings.BASE_DIR / "eda" / "chat_eval.md"
COMPARED = ("category", "filters", "count", "avoid_8am", "avoid_day", "branch")


def reference(student: Student, ideal: list[dict]) -> list[str]:
    """Courses the pipeline returns for the ideal calls, in order."""
    ctx = build_context(student)
    codes = []
    for call in ideal:
        result, _ = run(ctx, call.get("category"), call.get("about"), call.get("filters"), call.get("avoid_8am"),
                        call.get("avoid_day"), None, call.get("count"), call.get("branch"))
        codes += [course["code"] for course in result["courses"] if course["code"] not in codes]
    return codes


def made_calls(debug: dict) -> list[dict]:
    """The get_eligible_courses calls, as the app read them (after validation: nulls dropped, "3" -> 3)."""
    return [validate_call(call["name"], call["args"])[0] for round_ in debug.get("rounds", []) for call in round_["calls"]
            if call["name"] == "get_eligible_courses"]


def matches(ideal: dict, actual: dict) -> bool:
    """Same category, filters, count and timetable preferences (missing = empty)."""
    def value(call: dict, key: str):
        found = call.get(key)
        if key == "filters" and found:
            found = {name: setting for name, setting in found.items() if setting is not False}
        return found or None
    return all(value(ideal, key) == value(actual, key) for key in COMPARED)


class Command(BaseCommand):
    help = "Run the chat cases through the real assistant and score tool calls and course recall."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--pause", type=float, default=3.0, help="seconds between cases (free-tier rate limits)")
        parser.add_argument("--only", type=int, nargs="*", help="case numbers to run (1-based)")

    def handle(self, *args, **options) -> None:
        cases = json.loads(CASES.read_text())
        chosen = options["only"] or range(1, len(cases) + 1)
        students = {student.user.username: student for student in Student.objects.select_related("user")}
        rows = []
        for number in chosen:
            case = cases[number - 1]
            student = students[case["student"]]
            wanted = reference(student, case["ideal"])
            started = time.monotonic()
            result = run_agent(student, case["message"])
            seconds = time.monotonic() - started
            calls = made_calls(result.debug)
            matched = sum(any(matches(ideal, actual) for actual in calls) for ideal in case["ideal"])
            carded = [card["code"] for card in result.cards]
            found = [code for code in wanted if code in carded]
            providers = sorted({r["usage"].get("provider", "?") for r in result.debug.get("rounds", []) if r.get("usage")})
            rows.append({"n": number, "case": case, "wanted": wanted, "carded": carded, "found": found, "calls": calls,
                         "matched": matched, "invented": bool(result.debug.get("guardrail")) and any(
                             action["action"] != "category reminder" for action in result.debug["guardrail"]),
                         "seconds": seconds, "providers": providers, "reply": result.reply,
                         "error": any("tool" in error for error in result.debug.get("errors", [])),
                         # why apart: no provider reachable (rate limits, network) says nothing about the answers
                         "unavailable": any("llm" in error for error in result.debug.get("errors", []))})
            self.stdout.write(f"{number:2}. calls {matched}/{len(case['ideal'])}  recall {len(found)}/{len(wanted)}  "
                              f"{seconds:4.1f}s  {case['message']}" + ("  (LLM unavailable)" if rows[-1]["unavailable"] else ""))
            time.sleep(options["pause"])
        self.write(rows)

    def write(self, all_rows: list[dict]) -> None:
        rows = [row for row in all_rows if not row["unavailable"]]
        skipped = [row["n"] for row in all_rows if row["unavailable"]]
        ideal_calls = sum(len(row["case"]["ideal"]) for row in rows)
        with_reference = [row for row in rows if row["wanted"]]
        empty = [row for row in rows if not row["wanted"]]
        lines = ["# Chat evaluation (real LLM)", "",
                 f"Generated by `python manage.py chat_eval` over `recommender/eda/chat_cases.json` ({len(rows)} cases). "
                 "The reference for each case is what the pipeline returns for the ideal tool call(s), so this measures the "
                 "LLM's part (understanding the message, calling the tools, presenting the results), not the ranking "
                 "(`embedding_eda` measures that without any LLM)."
                 + (f" Left out, no LLM reachable: cases {', '.join(map(str, skipped))}." if skipped else ""), "",
                 f"- Ideal get_eligible_courses calls made with the right category / filters / count / preferences: "
                 f"**{sum(row['matched'] for row in rows)}/{ideal_calls}**",
                 f"- Reference courses that reached the cards: **{sum(len(r['found']) for r in with_reference)}/"
                 f"{sum(len(r['wanted']) for r in with_reference)}** (cases with a reference: {len(with_reference)})",
                 f"- Cases with nothing to recommend that showed no cards: **{sum(not r['carded'] for r in empty)}/{len(empty)}**",
                 f"- Replies where the guardrail removed an invented course: **{sum(r['invented'] for r in rows)}**; "
                 f"tool errors: **{sum(r['error'] for r in rows)}**",
                 f"- Median seconds per message: **{sorted(r['seconds'] for r in rows)[len(rows) // 2]:.1f}**", "",
                 "| # | student | message | calls ok | reference | on cards | provider |", "|---|---|---|---|---|---|---|"]
        for row in rows:
            case = row["case"]
            lines.append(f"| {row['n']} | {case['student']} | {case['message']} | {row['matched']}/{len(case['ideal'])} | "
                         f"{', '.join(row['wanted']) or 'none'} | {', '.join(row['carded']) or 'none'} | {', '.join(row['providers'])} |")
        lines += ["", "## Calls the LLM made that differ from the ideal", ""]
        for row in rows:
            if row["matched"] < len(row["case"]["ideal"]):
                lines.append(f"- **{row['n']}. {row['case']['message']}**: ideal `{json.dumps(row['case']['ideal'])}`, "
                             f"made `{json.dumps(row['calls'])}`")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n")
        self.stdout.write("\n".join(lines[4:10]) + f"\n\nReport: {OUT}")
