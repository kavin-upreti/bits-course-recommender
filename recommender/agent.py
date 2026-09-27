"""One chat message -> reply + course cards (todo.md section 10). No history: every message starts fresh.

The LLM only picks tools and arguments and writes the reply; every academic decision is made by the tools.
Nothing about the student is sent to the LLM.
"""
import logging
import re
import time
import traceback
from dataclasses import dataclass, field

from students.models import Student

from . import config, guardrail, llm, tools
from .cards import build_cards
from .codes import find_codes
from .context import StudentContext, build_context
from .prompts import SYSTEM_PROMPT
from .tool_schemas import TOOL_SCHEMAS
from .validation import validate_call

logger = logging.getLogger(__name__)
TOOLS = {
    "get_remaining_requirements": tools.get_remaining_requirements,
    "get_eligible_courses": tools.get_eligible_courses,
    "check_plan": tools.check_plan,
    "get_course_details": tools.get_course_details,
}
EMPTY = "Please type a question."
TOO_LONG = "That message is too long. Please keep it under {limit} characters."
BUSY = "The assistant is busy right now (free-tier limit). Please try again in a minute."
FAILED = "Something went wrong while contacting the assistant. Please try again."
ANSWER_NOW = "Answer the student now using only the results above. Do not call any tools."
REWRITE = "Your reply mentions {codes}, which did not appear in any tool result. Rewrite your reply using only courses from the tool results."
SEARCH_MISSING = ("The student asked for {what}, but you haven't searched that category yet. Call get_eligible_courses "
                  "with category {first} (with the topic the student gave for it, if any), then answer.")
# categories a message can name; a search with no category doesn't count, since its topic was meant for the others
CATEGORY_WORDS = {"HUEL": r"\bhuels?\b|\bhumanit\w* electives?", "DEL": r"\bdels?\b|\bdiscipline electives?",
                  "OPEL": r"\bopels?\b|\bopen electives?"}
RESULT_LISTS = ("courses", "couldnt_verify", "excluded", "loosely_related")  # only "courses" can get cards unasked


@dataclass
class AgentResult:
    reply: str
    cards: list[dict]
    debug: dict      # rounds, calls (name, args, result size in chars), usage, guardrail actions, timings


@dataclass
class Run:
    """What one message's tool calls produced so far."""

    ctx: StudentContext
    seen_codes: set[str] = field(default_factory=set)
    listed: dict[str, dict] = field(default_factory=dict)     # code -> latest entry of a `courses` list
    mentioned: dict[str, dict] = field(default_factory=dict)  # code -> latest entry of any result list
    plans: list[dict] = field(default_factory=list)           # successful check_plan results, in order
    plan_calls: int = 0
    excluded: dict[str, dict] = field(default_factory=dict)     # code -> latest excluded entry
    not_offered: dict[str, dict] = field(default_factory=dict)  # code -> better match not offered this semester
    shortfalls: list[str] = field(default_factory=list)          # "Only 1 of the 3 courses you asked for…"
    debug: dict = field(default_factory=lambda: {"rounds": [], "guardrail": [], "errors": []})
    named: set[str] = field(default_factory=set)       # categories the student's message asks for
    searched: set[str] = field(default_factory=set)    # categories a get_eligible_courses call named explicitly
    reminded: bool = False


def named_categories(message: str) -> set[str]:
    return {category for category, pattern in CATEGORY_WORDS.items() if re.search(pattern, message, re.I)}


def codes_in(value) -> set[str]:
    """Every course code anywhere in a tool result (keys and values)."""
    if isinstance(value, dict):
        return set().union(*(codes_in(key) | codes_in(item) for key, item in value.items())) if value else set()
    if isinstance(value, list):
        return set().union(*(codes_in(item) for item in value)) if value else set()
    return set(find_codes(value)) if isinstance(value, str) else set()


def record(run: Run, name: str, result: dict) -> None:
    """Remember the codes, course entries and successful plans a result contains."""
    run.seen_codes |= codes_in(result)
    for key in RESULT_LISTS:
        for entry in result.get(key, []):
            run.mentioned[entry["code"]] = entry
            if key == "courses":
                run.listed[entry["code"]] = entry
    if name == "check_plan" and result.get("ok"):
        run.plans.append(result)
    run.excluded.update({entry["code"]: entry for entry in result.get("excluded", [])})
    run.not_offered.update({entry["code"]: entry for entry in result.get("better_matches_not_offered", [])})
    if result.get("shortfall") and result["shortfall"] not in run.shortfalls:
        run.shortfalls.append(result["shortfall"])


def missing_notes(run: Run, reply: str) -> str:
    """What the reply should have said but didn't: courses left out (and why) and better matches not offered.
    why in Python: models often skip these, and the student needs them to trust the list."""
    mentioned = set(find_codes(reply))
    lines = [f"- {text}" for text in run.shortfalls]
    lines += [f"- {entry['code']} {entry['title']} was left out: {entry['reason']}"
              for code, entry in run.excluded.items() if code not in mentioned]
    unoffered = [f"{entry['code']} {entry['title']}" for code, entry in run.not_offered.items() if code not in mentioned]
    if unoffered:
        lines.append(f"- Better matches that aren't offered this semester: {', '.join(unoffered)}")
    return "\n\n**Also:**\n" + "\n".join(lines) if lines else ""


def without_card_bullets(reply: str, cards: list[dict]) -> str:
    """Drop the reply's bullet lines about courses that have a card: the card already shows all of it."""
    carded = {card["code"] for card in cards}
    kept = [line for line in reply.splitlines()
            if not (line.lstrip().startswith(("- ", "* ", "• ")) and carded & set(find_codes(line)))]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def execute(run: Run, call: llm.ToolCall) -> dict:
    """Validate, then run one tool; errors come back as results the LLM can read."""
    args, error = validate_call(call.name, call.args)
    if error:
        return {"error": error}
    if call.name == "check_plan":
        run.plan_calls += 1
        if run.plan_calls > config.MAX_CHECK_PLAN_CALLS:
            return {"error": "check_plan limit reached for this message. Answer with the results you have."}
    try:
        return TOOLS[call.name](run.ctx, **args)
    except Exception:  # why broad: one broken tool call must not end the chat; the LLM gets told and carries on
        run.debug["errors"].append({"tool": call.name, "traceback": traceback.format_exc()})
        logger.exception("Tool %s failed", call.name)
        return {"error": f"Internal error in {call.name}"}


def tool_round(run: Run, messages: list[dict], response: llm.LLMResponse, number: int) -> None:
    """Run the response's tool calls in order and append the assistant and tool turns."""
    messages.append({"role": "assistant", "text": response.text, "tool_calls": response.tool_calls, "raw": response.raw})
    results, calls = [], []
    for call in response.tool_calls:
        started, timed = time.monotonic(), len(run.ctx.timings)
        result = execute(run, call)
        record(run, call.name, result)
        if call.name == "get_eligible_courses" and isinstance(call.args, dict) and isinstance(call.args.get("category"), str):
            run.searched.add(call.args["category"].strip().upper())
        results.append({"id": call.id, "name": call.name, "result": result})
        calls.append({"name": call.name, "args": call.args, "result_chars": len(str(result)),
                      "ms": round(1000 * (time.monotonic() - started)),
                      **({"stage_c": run.ctx.timings[timed]} if len(run.ctx.timings) > timed else {})})
    messages.append({"role": "tool", "results": results})
    run.debug["rounds"].append({"round": number, "calls": calls, "usage": response.usage})


def loop(run: Run, messages: list[dict]) -> str:
    """Up to MAX_AGENT_ROUNDS tool rounds, then a forced text answer."""
    for number in range(1, config.MAX_AGENT_ROUNDS + 1):
        response = llm.chat(SYSTEM_PROMPT, messages, TOOL_SCHEMAS)
        if not response.tool_calls:
            run.debug["rounds"].append({"round": number, "calls": [], "usage": response.usage})
            missing = sorted(run.named - run.searched)
            if not missing or run.reminded:
                return response.text or ""
            # why in Python: "courses on AI, and a HUEL on media" was answered from one no-category search, so the
            # media topic was never searched among HUELs. One reminder, then whatever the model says goes.
            run.reminded = True
            run.debug["guardrail"].append({"action": "category reminder", "categories": missing})
            messages += [{"role": "assistant", "text": response.text or "", "tool_calls": []},
                         {"role": "user", "text": SEARCH_MISSING.format(what=" and ".join(f"a {c}" for c in missing), first=missing[0])}]
            continue
        tool_round(run, messages, response, number)
    messages.append({"role": "user", "text": ANSWER_NOW})
    response = llm.chat(SYSTEM_PROMPT, messages, None)
    run.debug["rounds"].append({"round": "final (no tools)", "calls": [], "usage": response.usage})
    return response.text or ""


def guard(run: Run, messages: list[dict], reply: str, student_codes: set[str]) -> str:
    """One rewrite if the reply names a course no tool returned; still wrong -> drop those sentences."""
    allowed = guardrail.allowed_codes(run.seen_codes, student_codes)
    bad = guardrail.disallowed(reply, allowed)
    if not bad:
        return reply
    run.debug["guardrail"].append({"action": "retry", "codes": bad})
    messages += [{"role": "assistant", "text": reply, "tool_calls": []},
                 {"role": "user", "text": REWRITE.format(codes=", ".join(bad))}]
    reply = llm.chat(SYSTEM_PROMPT, messages, None).text or ""
    bad = guardrail.disallowed(reply, allowed)
    if bad:
        run.debug["guardrail"].append({"action": "removed sentences", "codes": bad})
        reply = guardrail.strip_codes(reply, bad)
    return reply


def run_agent(student: Student, message: str) -> AgentResult:
    """Answer one message: tool loop, guardrail, cards."""
    message = (message or "").strip()
    if not message:
        return AgentResult(EMPTY, [], {})
    if len(message) > config.MAX_MESSAGE_CHARS:
        return AgentResult(TOO_LONG.format(limit=config.MAX_MESSAGE_CHARS), [], {})
    started = time.monotonic()
    run = Run(build_context(student), named=named_categories(message))
    messages = [{"role": "user", "text": message}]
    try:
        reply = guard(run, messages, loop(run, messages), set(find_codes(message.upper())))
    except llm.LLMUnavailable as error:
        run.debug["errors"].append({"llm": f"unavailable: {error}"})
        return AgentResult(BUSY, [], run.debug)
    except llm.LLMError as error:
        run.debug["errors"].append({"llm": str(error)})
        return AgentResult(FAILED, [], run.debug)
    cards = build_cards(run.ctx, reply, run.plans, run.listed, run.mentioned)
    # after the cards: these courses aren't recommendations; notes read the full reply, before its bullets go
    reply = without_card_bullets(reply, cards) + missing_notes(run, reply)
    run.debug["seconds"] = round(time.monotonic() - started, 2)
    run.debug["usage"] = {key: sum(r["usage"].get(key, 0) for r in run.debug["rounds"]) for key in ("input_tokens", "output_tokens")}
    return AgentResult(reply, cards, run.debug)
