"""Provider fallback in llm.chat and the OpenAI-format conversion (no network: _send is patched)."""
import json
from unittest.mock import patch

from django.test import SimpleTestCase

from recommender import llm
from recommender.llm import LLMError, LLMResponse, LLMUnavailable, ProviderBusy, ToolCall

ENV = {"LLM_PROVIDERS": "groq:m1,openrouter:vendor/m2:free,gemini:m3",
       "GROQ_API_KEY": "k", "OPENROUTER_API_KEY": "k", "GEMINI_API_KEY": "k"}


class FallbackTests(SimpleTestCase):
    def run_chat(self, outcomes: dict, env: dict = ENV):
        """outcomes: provider -> list of responses / exceptions, consumed per call."""
        tried = []

        def send(provider, model, system, messages, tools):
            tried.append((provider, model))
            outcome = outcomes[provider].pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch.dict("os.environ", env, clear=True), patch.object(llm, "_send", send), patch("time.sleep") as sleep:
            try:
                return llm.chat("sys", [], None), tried, sleep
            except (LLMError, LLMUnavailable) as error:
                return error, tried, sleep

    def test_parses_providers_with_colons_in_model(self):
        with patch.dict("os.environ", ENV, clear=True):
            self.assertEqual(llm.providers(), [("groq", "m1"), ("openrouter", "vendor/m2:free"), ("gemini", "m3")])

    def test_first_provider_answers(self):
        response, tried, _ = self.run_chat({"groq": [LLMResponse("hi", [])]})
        self.assertEqual((response.text, response.usage["provider"], tried), ("hi", "groq:m1", [("groq", "m1")]))

    def test_busy_and_broken_providers_fall_through(self):
        response, tried, sleep = self.run_chat({"groq": [ProviderBusy("429")], "openrouter": [LLMError("400")],
                                                "gemini": [LLMResponse("from gemini", [])]})
        self.assertEqual((response.text, [name for name, _ in tried]), ("from gemini", ["groq", "openrouter", "gemini"]))
        sleep.assert_not_called()

    def test_all_busy_retries_then_unavailable(self):
        busy = lambda: [ProviderBusy("429")] * 4  # noqa: E731
        error, tried, sleep = self.run_chat({"groq": busy(), "openrouter": busy(), "gemini": busy()})
        self.assertIsInstance(error, LLMUnavailable)
        self.assertEqual(len(tried), 12)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4, 8])

    def test_busy_provider_recovers_after_backoff(self):
        response, tried, _ = self.run_chat({"groq": [ProviderBusy("429"), LLMResponse("later", [])],
                                            "openrouter": [LLMError("401")], "gemini": [LLMError("404")]})
        self.assertEqual(response.text, "later")
        self.assertEqual([name for name, _ in tried], ["groq", "openrouter", "gemini", "groq"])

    def test_all_errors_raise_llm_error(self):
        error, _, _ = self.run_chat({"groq": [LLMError("401 bad key")], "openrouter": [LLMError("x")], "gemini": [LLMError("y")]})
        self.assertIsInstance(error, LLMError)
        self.assertIn("401 bad key", str(error))

    def test_providers_without_keys_are_skipped(self):
        response, tried, _ = self.run_chat({"gemini": [LLMResponse("ok", [])]},
                                           {"LLM_PROVIDERS": ENV["LLM_PROVIDERS"], "GEMINI_API_KEY": "k"})
        self.assertEqual(tried, [("gemini", "m3")])
        error, _, _ = self.run_chat({}, {"LLM_PROVIDERS": ENV["LLM_PROVIDERS"]})
        self.assertIn("No LLM API key", str(error))


class OpenAIFormatTests(SimpleTestCase):
    def test_messages(self):
        messages = [
            {"role": "user", "text": "hi"},
            {"role": "assistant", "text": None, "tool_calls": [ToolCall("c1", "check_plan", {"courses": ["A B101"]})], "raw": None},
            {"role": "tool", "results": [{"id": "c1", "name": "check_plan", "result": {"ok": True}}]},
        ]
        converted = llm._openai_messages("groq", "sys", messages)
        self.assertEqual(converted[0], {"role": "system", "content": "sys"})
        self.assertEqual(converted[2]["tool_calls"][0]["function"], {"name": "check_plan", "arguments": json.dumps({"courses": ["A B101"]})})
        self.assertEqual(converted[3], {"role": "tool", "tool_call_id": "c1", "content": '{"ok":true}'})

    def test_gemini_gets_its_own_turns_back_and_a_placeholder_signature_for_others(self):
        signed = {"role": "assistant", "tool_calls": [{"id": "g1", "extra_content": {"google": {"thought_signature": "sig"}}}]}
        messages = [{"role": "user", "text": "hi"},
                    {"role": "assistant", "text": None, "tool_calls": [ToolCall("c1", "check_plan", {})], "raw": ("groq", {})},
                    {"role": "assistant", "text": None, "tool_calls": [ToolCall("g1", "check_plan", {})], "raw": ("gemini", signed)}]
        converted = llm._openai_messages("gemini", "sys", messages)
        self.assertEqual(converted[2]["tool_calls"][0]["extra_content"], llm.SKIP_SIGNATURE)
        self.assertIs(converted[3], signed)
        self.assertNotIn("extra_content", llm._openai_messages("groq", "sys", messages)[2]["tool_calls"][0])

    def test_a_200_without_choices_is_busy_not_a_crash(self):
        class Response:
            status_code, text = 200, '{"error": {"code": 429}}'

            def json(self):
                return {"error": {"code": 429}}
        with patch.dict("os.environ", ENV), patch("httpx.post", return_value=Response()):
            with self.assertRaises(ProviderBusy):
                llm._send("openrouter", "m", "sys", [{"role": "user", "text": "hi"}], None)

    def test_bad_argument_json_reaches_validation(self):
        self.assertEqual(llm._parse_args("{oops"), {"unparseable_arguments": "{oops"})
        self.assertEqual(llm._parse_args(""), {})


class NullableSchemaTests(SimpleTestCase):
    def test_optional_properties_accept_null(self):
        from recommender.tool_schemas import TOOL_SCHEMAS
        eligible = llm._nullable(TOOL_SCHEMAS[1]["parameters"])
        self.assertEqual(eligible["properties"]["avoid_8am"]["type"], ["boolean", "null"])
        self.assertEqual(eligible["properties"]["category"]["enum"], ["HUEL", "DEL", "OPEL", None])
        self.assertEqual(eligible["properties"]["filters"]["properties"]["max_quizzes"]["type"], ["integer", "null"])
        plan = llm._nullable(TOOL_SCHEMAS[2]["parameters"])
        self.assertEqual(plan["properties"]["courses"]["type"], "array")  # required: unchanged
        self.assertEqual(TOOL_SCHEMAS[1]["parameters"]["properties"]["avoid_8am"]["type"], "boolean")  # original untouched
