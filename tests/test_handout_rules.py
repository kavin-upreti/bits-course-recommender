"""Checks for the rule-based parsing in extractors/handouts.py (no PDFs or models needed).

Run: .venv/bin/python -m pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extractors"))

import handouts as h  # noqa: E402


def test_parse_weight():
    assert h.parse_weight("25%") == 25
    assert h.parse_weight("50 (25 %)") == 25  # a percent beats marks
    assert h.parse_weight("–") is None


def test_headings():
    assert h.heading_of("1. Course Description:")[0] == "description"
    assert h.heading_of("5. Evaluation Scheme:")[0] == "evaluation"
    assert h.heading_of("Make-up Policy: No make-up for quizzes.") == ("makeup", "No make-up for quizzes.")
    assert h.heading_of("The students will learn this.") is None


def test_component_kinds_need_word_start():
    assert h.component("Evaluative Tutorials", 10, "", "", "")["kind"] == "tutorial_test"
    assert h.component("Mid-Semester Test", 30, "90 min", "", "Closed Book")["nature"] == "CB"


def test_printed_code_check():
    titles = {"BIO U101": "INTRO TO BIO SCIENCES", "BIO G523": "ADV & APPLIED MICROBIO"}
    assert h.printed_code_issue("025_BIO_U101.pdf", "BIO U101", "BIO F101", "Introduction to Biological Sciences", titles)[1] is False
    assert h.printed_code_issue("017_BIO_G523.pdf", "BIO G523", "BIO F212", "Microbiology", titles)[1] is True
