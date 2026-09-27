"""validate_call (todo.md 9.3) and the schemas loading into the Gemini wrapper."""
from django.test import SimpleTestCase

from recommender.llm import _to_tools
from recommender.tool_schemas import TOOL_SCHEMAS
from recommender.validation import validate_call


class ValidationTests(SimpleTestCase):
    def test_clean_call(self):
        args = {"category": "DEL", "about": "robots", "filters": {"no_midsem": True, "max_quizzes": 3.0},
                "avoid_8am": None, "avoid_day": "", "exclude": ["CS F111"]}
        self.assertEqual(validate_call("get_eligible_courses", args),
                         ({"category": "DEL", "about": ["robots"], "filters": {"no_midsem": True, "max_quizzes": 3},
                           "exclude": ["CS F111"]}, None))
        self.assertEqual(validate_call("get_remaining_requirements", None), ({}, None))

    def test_text_none_and_booleans_from_sloppy_models(self):
        args = {"category": "DEL", "about": "machine learning", "filters": "None", "exclude": "null", "avoid_day": "None",
                "avoid_8am": "False"}
        self.assertEqual(validate_call("get_eligible_courses", args),
                         ({"category": "DEL", "about": ["machine learning"], "avoid_8am": False}, None))
        self.assertEqual(validate_call("check_plan", {"courses": ["A B101"], "avoid_8am": "TRUE"}),
                         ({"courses": ["A B101"], "avoid_8am": True}, None))

    def test_errors(self):
        cases = [
            ("nope", {}, "Unknown tool nope. Available: get_remaining_requirements, get_eligible_courses, check_plan, get_course_details."),
            ("get_eligible_courses", {"topic": "x"}, "Unknown argument topic. Allowed: category, about, filters, count, avoid_8am, avoid_day, exclude."),
            ("get_eligible_courses", {"category": "CDC"}, "category must be one of: HUEL, DEL, OPEL."),
            ("get_eligible_courses", {"about": ["x" * 101]}, "each item of about is too long (at most 100 characters)."),
            ("get_eligible_courses", {"about": ["a"] * 9}, "about must have between 0 and 8 items."),
            ("get_eligible_courses", {"avoid_8am": "yes"}, "avoid_8am must be true or false."),
            ("get_eligible_courses", {"avoid_day": "Su"}, "avoid_day must be one of: M, T, W, Th, F, S."),
            ("get_eligible_courses", {"filters": {"easy": True}}, "Unknown filter easy. Allowed: no_midsem, no_attendance, lenient_makeup, project_based, open_book, max_quizzes, max_compre_percent, min_project_percent."),
            ("get_eligible_courses", {"filters": {"max_quizzes": 21}}, "max_quizzes must be between 0 and 20."),
            ("get_eligible_courses", {"filters": {"max_quizzes": 2.5}}, "max_quizzes must be a whole number."),
            ("get_eligible_courses", {"filters": "no midsem"}, "filters must be an object."),
            ("get_eligible_courses", {"exclude": ["A"] * 21}, "exclude must have between 0 and 20 items."),
            ("get_eligible_courses", {"exclude": [3]}, "each item of exclude must be a string."),
            ("check_plan", {}, "Missing required argument: courses."),
            ("check_plan", {"courses": []}, "courses must have between 1 and 10 items."),
            ("get_course_details", {"code": 7}, "code must be a string."),
        ]
        for name, args, error in cases:
            with self.subTest(args=args):
                self.assertEqual(validate_call(name, args), ({}, error))

    def test_schemas_load_in_wrapper(self):
        tools = _to_tools(TOOL_SCHEMAS)
        self.assertEqual([declaration.name for declaration in tools[0].function_declarations],
                         [tool["name"] for tool in TOOL_SCHEMAS])


class NoNetworkTests(SimpleTestCase):
    def test_network_is_blocked_and_embedder_is_fake(self):
        import socket

        from recommender.embeddings import get_embedder
        from recommender.testing import FakeEmbedder, NetworkUsedInTest
        with self.assertRaises(NetworkUsedInTest), socket.socket() as connection:
            connection.connect(("example.com", 80))
        self.assertIsInstance(get_embedder(), FakeEmbedder)
