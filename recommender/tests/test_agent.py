"""run_agent with a scripted FakeLLM (todo.md 10.4, tests 1-12). No network."""
import json
from unittest.mock import patch

from recommender import config
from recommender.agent import BUSY, run_agent
from recommender.guardrail import SORRY
from recommender.llm import LLMResponse, LLMUnavailable, ToolCall
from recommender.prompts import SYSTEM_PROMPT
from recommender.testing import FakeLLM
from recommender.tool_schemas import TOOL_SCHEMAS

from .fixtures import offer
from .test_eligible import Catalog


def calls(*pairs) -> LLMResponse:
    return LLMResponse(text=None, tool_calls=[ToolCall(f"c{n}", name, args) for n, (name, args) in enumerate(pairs)])


def text(reply: str) -> LLMResponse:
    return LLMResponse(text=reply, tool_calls=[])


class AgentTests(Catalog):
    def setUp(self) -> None:
        super().setUp()
        # XX F211 (current) on M; XX F411 and HSS F201 clash on T; HSS F202 on W
        for code, timings in (("XX F211", {"M": [2]}), ("XX F411", {"T": [3]}), ("HSS F201", {"T": [3]}), ("HSS F202", {"W": [3]})):
            self.courses[code].offerings.all().delete()
            offer(self.courses[code], [("lecture", "L1", timings)])
        self.student.interests = ["quantum knitting"]
        self.student.save()

    def agent(self, responses: list, message: str = "give me good courses related to LLMs, no 8 am"):
        fake = FakeLLM(responses)
        with patch("recommender.llm.chat", fake):
            result = run_agent(self.student, message)
        return result, fake

    def tool_results(self, fake: FakeLLM, request: int) -> list[dict]:
        return fake.requests[request]["messages"][-1]["results"]

    def test_01_main_example(self):
        result, fake = self.agent([
            calls(("get_eligible_courses", {"category": "HUEL", "avoid_8am": True}),
                  ("get_eligible_courses", {"category": "DEL", "about": "large language models", "avoid_8am": True})),
            calls(("check_plan", {"courses": ["HSS F201", "XX F411"], "avoid_8am": True})),
            calls(("check_plan", {"courses": ["HSS F202", "XX F411"], "avoid_8am": True})),
            text("HSS F202 Film Studies (HUEL), lecture L1.\nXX F411 Natural Language Processing (DEL), lecture L1."),
        ])
        self.assertEqual(len(fake.requests), 4)
        first = self.tool_results(fake, 1)
        self.assertEqual([(r["id"], r["result"]["searched"]) for r in first], [("c0", ["HUEL"]), ("c1", ["DEL"])])
        self.assertFalse(self.tool_results(fake, 2)[0]["result"]["ok"])
        self.assertTrue(self.tool_results(fake, 3)[0]["result"]["ok"])
        self.assertTrue(result.reply.startswith("HSS F202 Film Studies"))
        self.assertEqual([(card["code"], card["sections"], card["category"]) for card in result.cards],
                         [("HSS F202", {"lecture": "L1"}, "HUEL"), ("XX F411", {"lecture": "L1"}, "DEL")])
        self.assertEqual(result.cards[0]["facts"][0], {"label": "Midsem", "value": "Couldn't verify (no handout)"})
        self.assertEqual(len(result.debug["rounds"]), 4)

    def test_02_question_back(self):
        result, fake = self.agent([text("Which courses do you mean by 'the second one'?")], "swap the second one")
        self.assertEqual((result.reply, result.cards, len(fake.requests)), ("Which courses do you mean by 'the second one'?", [], 1))

    def test_03_invalid_args_fixed(self):
        result, fake = self.agent([
            calls(("get_eligible_courses", {"category": "Humanities"})),
            calls(("get_eligible_courses", {"category": "HUEL"})),
            text("Try HSS F201 Introductory Psychology."),
        ])
        self.assertEqual(self.tool_results(fake, 1)[0]["result"], {"error": "category must be one of: HUEL, DEL, OPEL."})
        self.assertEqual(self.tool_results(fake, 2)[0]["result"]["searched"], ["HUEL"])
        self.assertEqual([card["code"] for card in result.cards], ["HSS F201"])  # no check_plan: codes from `courses`
        self.assertEqual(result.cards[0]["sections"], {"lecture": "L1"})  # picked by get_eligible_courses' fit check

    def test_04_unknown_tool(self):
        _, fake = self.agent([calls(("add_to_plan", {"code": "XX F411"})), text("Done.")])
        self.assertTrue(self.tool_results(fake, 1)[0]["result"]["error"].startswith("Unknown tool add_to_plan."))

    def test_05_tool_exception(self):
        with patch("recommender.agent.TOOLS", {"get_remaining_requirements": lambda ctx: 1 / 0}):
            result, fake = self.agent([calls(("get_remaining_requirements", {})), text("Sorry, try later.")])
        self.assertEqual(self.tool_results(fake, 1)[0]["result"], {"error": "Internal error in get_remaining_requirements"})
        self.assertIn("ZeroDivisionError", result.debug["errors"][0]["traceback"])
        self.assertEqual(result.reply, "Sorry, try later.")

    def test_06_round_cap(self):
        responses = [calls(("get_remaining_requirements", {}))] * config.MAX_AGENT_ROUNDS + [text("You need 2 DELs.")]
        result, fake = self.agent(responses)
        self.assertEqual(len(fake.requests), config.MAX_AGENT_ROUNDS + 1)
        final = fake.requests[-1]
        self.assertIsNone(final["tools"])
        self.assertEqual(final["messages"][-1], {"role": "user", "text": "Answer the student now using only the results above. Do not call any tools."})
        self.assertEqual(result.reply, "You need 2 DELs.")

    def test_07_guardrail_retry_fixes(self):
        result, fake = self.agent([
            calls(("get_eligible_courses", {"category": "DEL"})),
            text("Take XX F411 and also ZZ F999."),
            text("Take XX F411."),
        ])
        self.assertEqual(result.reply, "Take XX F411.")
        self.assertIn("ZZ F999, which did not appear", fake.requests[2]["messages"][-1]["text"])
        self.assertIsNone(fake.requests[2]["tools"])
        self.assertEqual(result.debug["guardrail"], [{"action": "retry", "codes": ["ZZ F999"]}])

    def test_08_guardrail_removes_sentence(self):
        result, _ = self.agent([
            calls(("get_eligible_courses", {"category": "DEL"})),
            text("Take XX F411. Also ZZ F999 is great."),
            text("Take XX F411. Really, ZZ F999 is great.\nZZ F998 too."),
        ])
        self.assertEqual(result.reply, "Take XX F411.")
        result, _ = self.agent([text("ZZ F999 is great."), text("ZZ F999 is great.")])
        self.assertEqual(result.reply, SORRY)

    def test_09_student_typed_code_allowed(self):
        result, fake = self.agent([text("QQ F123 isn't something I know about.")], "tell me about qq f123")
        self.assertEqual((result.reply, len(fake.requests)), ("QQ F123 isn't something I know about.", 1))

    def test_10_llm_unavailable(self):
        result, _ = self.agent([LLMUnavailable("429")])
        self.assertEqual((result.reply, result.cards), (BUSY, []))

    def test_11_nothing_about_the_student_is_sent(self):
        _, fake = self.agent([
            calls(("get_eligible_courses", {}), ("get_remaining_requirements", {}), ("check_plan", {"courses": ["XX F411"]})),
            text("ok"),
        ])
        sent = json.dumps(fake.requests, default=str)
        for secret in ("Asha", "Testname", "B.E. Test", "quantum knitting", self.student.user.username, "2024"):
            self.assertNotIn(secret, sent)

    def test_12_system_prompt_and_schemas_unchanged(self):
        _, fake = self.agent([calls(("get_remaining_requirements", {})), text("ok")])
        for request in fake.requests:
            self.assertEqual((request["system"], request["tools"]), (SYSTEM_PROMPT, TOOL_SCHEMAS))

    def test_check_plan_calls_are_capped(self):
        plan = ("check_plan", {"courses": ["XX F411"]})
        _, fake = self.agent([calls(plan, plan, plan, plan), text("ok")])
        results = [r["result"] for r in self.tool_results(fake, 1)]
        self.assertTrue(all("ok" in result for result in results[:config.MAX_CHECK_PLAN_CALLS]))
        self.assertEqual(results[-1], {"error": "check_plan limit reached for this message. Answer with the results you have."})

    def test_unmentioned_exclusions_are_added(self):
        self.courses["XX F412"].offerings.all().delete()
        offer(self.courses["XX F412"], [("lecture", "L1", {"M": [2]})])  # clashes with the current XX F211
        result, _ = self.agent([calls(("get_eligible_courses", {"category": "DEL"})), text("Take **XX F411**.")])
        self.assertTrue(result.reply.startswith("Take **XX F411**.\n\n**Also:**\n- XX F412 Deep Learning was left out: "))
        self.assertEqual([card["code"] for card in result.cards], ["XX F411"])  # the note adds no card
        result, _ = self.agent([calls(("get_eligible_courses", {"category": "DEL"})), text("XX F411; XX F412 clashes.")])
        self.assertNotIn("Also", result.reply)  # already mentioned

    def test_named_category_that_was_never_searched_gets_one_reminder(self):
        message = "courses on LLMs, and a HUEL on psychology"
        result, fake = self.agent([calls(("get_eligible_courses", {"about": ["large language models"]})), text("Here."),
                                   calls(("get_eligible_courses", {"category": "HUEL", "about": ["psychology"]})), text("Done.")], message)
        self.assertEqual(len(fake.requests), 4)
        self.assertIn("haven't searched that category yet. Call get_eligible_courses with category HUEL", fake.requests[2]["messages"][-1]["text"])
        self.assertEqual(result.debug["guardrail"], [{"action": "category reminder", "categories": ["HUEL"]}])
        # searched explicitly (or reminded once already): no reminder
        result, fake = self.agent([calls(("get_eligible_courses", {"category": "HUEL"})), text("Here.")], message)
        self.assertEqual(len(fake.requests), 2)
        result, fake = self.agent([text("Here."), text("Still here.")], message)
        self.assertEqual((len(fake.requests), result.reply), (2, "Still here."))

    def test_empty_and_long_messages(self):
        self.assertEqual(run_agent(self.student, "   ").reply, "Please type a question.")
        self.assertIn("too long", run_agent(self.student, "x" * (config.MAX_MESSAGE_CHARS + 1)).reply)


class EndpointTests(Catalog):
    def setUp(self) -> None:
        super().setUp()
        self.client.force_login(self.student.user)

    def post(self, body) -> object:
        return self.client.post("/recommend/", body if isinstance(body, str) else json.dumps(body), content_type="application/json")

    def test_reply_cards_and_debug_only_in_debug_mode(self):
        with patch("recommender.llm.chat", FakeLLM([text("Hello")])), self.settings(DEBUG=True):
            body = self.post({"message": "hi"}).json()
        self.assertEqual((body["reply"], body["cards"]), ("Hello", []))
        self.assertIn("rounds", body["debug"])
        with patch("recommender.llm.chat", FakeLLM([text("Hello")])), self.settings(DEBUG=False):
            self.assertNotIn("debug", self.post({"message": "hi"}).json())

    def test_bad_requests(self):
        self.assertEqual(self.post("not json").status_code, 400)
        self.assertEqual(self.post({"text": "hi"}).status_code, 400)
        self.assertEqual(self.client.get("/recommend/").status_code, 405)

    def test_no_profile_and_login(self):
        from django.contrib.auth.models import User
        self.client.force_login(User.objects.create_user("nobody", password="x"))
        response = self.post({"message": "hi"})
        self.assertEqual((response.status_code, response.json()), (400, {"error": "Please create your profile first."}))
        self.client.logout()
        self.assertEqual(self.post({"message": "hi"}).status_code, 302)

    def test_chat_page(self):
        page = self.client.get("/chat/")
        self.assertContains(page, "Suggest DELs related to AI")
        self.assertContains(page, "Each question is answered on its own.")


class PickingTests(Catalog):
    """Select on the chat page -> preview timetables -> finalise -> remove on the semester page."""

    def setUp(self) -> None:
        super().setUp()
        from catalog.models import Rule
        Rule.objects.create(rule_id="tt_lunch_hour", group="timetable", description="", values={"lunch_periods": [4, 5, 6]})
        Rule.objects.create(rule_id="tt_periods", group="timetable", description="",
                            values=[{"period": n, "start": f"{7 + n:02d}:00", "end": f"{7 + n:02d}:50"} for n in range(1, 11)])
        for code, timings in (("XX F411", {"W": [3]}), ("HSS F202", {"F": [3]})):
            self.courses[code].offerings.all().delete()
            offer(self.courses[code], [("lecture", "L1", timings)])
        from students.models import StudentCourse
        StudentCourse.objects.filter(course__code="XX F211").update(source="pattern")  # from the chart, like real rows
        self.client.force_login(self.student.user)

    def select(self, code: str, selected: bool = True):
        return self.client.post("/plan/select/", json.dumps({"code": code, "selected": selected}), content_type="application/json")

    def test_select_and_unselect(self):
        body = self.select("xx f411").json()
        self.assertEqual(body["selection"], [{"code": "XX F411", "title": "Natural Language Processing", "category": "DEL"}])
        self.assertEqual(self.select("XX F411", False).json()["selection"], [])
        self.assertEqual(self.select("XX F211").json()["error"], "XX F211 (or an equivalent) is already done or in this semester.")
        self.assertEqual(self.select("NOPE F999").status_code, 400)
        self.assertContains(self.client.get("/chat/"), "Selected courses")

    def test_selection_is_capped(self):
        with patch.object(config, "MAX_PLAN_COURSES", 1):
            self.select("XX F411")
            self.assertIn("at most 1", self.select("XX F412").json()["error"])

    def test_clashing_course_is_refused_on_select_and_finalise(self):
        from students.models import StudentCourse
        self.courses["HSS F202"].offerings.all().delete()
        offer(self.courses["HSS F202"], [("lecture", "L1", {"W": [3]})])  # same slot as XX F411's only lecture
        self.select("XX F411")
        refused = self.select("HSS F202")
        self.assertEqual(refused.status_code, 400)
        self.assertIn("Can't add HSS F202", refused.json()["error"])
        self.assertEqual([item["code"] for item in self.select("XX F411").json()["selection"]], ["XX F411"])
        both = self.client.post("/plan/finalise/", json.dumps({"codes": ["XX F411", "HSS F202"]}), content_type="application/json")
        self.assertEqual(both.status_code, 400)  # the selection is re-checked, not trusted
        self.assertIn("can't all be taken together", both.json()["error"])
        self.assertFalse(StudentCourse.objects.filter(course__code__in=["XX F411", "HSS F202"]).exists())

    def test_timetables_popup_includes_current_selected_and_candidate(self):
        self.select("XX F411")
        page = self.client.get("/plan/timetables/?with=HSS%20F202").content.decode()
        self.assertIn("Timetables with XX F411, HSS F202", page)
        self.assertIn("XX F211", page)            # current course is in the grid
        self.assertIn('t-lecture new"', page)     # considered courses are drawn dashed

    def test_finalise_and_remove(self):
        from students.models import StudentCourse
        self.select("XX F411")
        response = self.client.post("/plan/finalise/", json.dumps({"codes": ["XX F411"]}), content_type="application/json")
        self.assertEqual(response.json(), {"redirect": "/semester/"})
        row = StudentCourse.objects.get(student=self.student, course__code="XX F411")
        self.assertEqual((row.status, row.source, row.category, row.semester_taken), ("current", "user", "DEL", "3-1"))
        self.assertEqual(self.client.session["plan_selection"], [])
        bad = self.client.post("/plan/finalise/", json.dumps({"codes": ["XX F411"]}), content_type="application/json")
        self.assertEqual(bad.status_code, 400)  # already in this semester

        self.client.post("/semester/remove/", {"code": "XX F211"})  # not added from here: stays
        self.assertTrue(StudentCourse.objects.filter(course__code="XX F211").exists())
        self.client.post("/semester/remove/", {"code": "XX F411"})
        self.assertFalse(StudentCourse.objects.filter(course__code="XX F411").exists())
