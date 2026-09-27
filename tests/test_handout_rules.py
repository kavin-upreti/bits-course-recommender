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


def test_page_furniture_dropped():
    header = ["BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, Pilani", "Pilani Campus"]
    lines = [(1, text) for text in header + ["COURSE HANDOUT", "1. Course Description: Gender studies.", "body", "1"]]
    lines += [(2, text) for text in header + ["more description", "4. Text Books", "Page 2 of 3"]]
    kept = [text for _, text in h.drop_page_furniture(lines)]
    assert kept == ["COURSE HANDOUT", "1. Course Description: Gender studies.", "body", "more description", "4. Text Books"]
    # a one-page handout keeps its letterhead lines (nothing repeats), only the page number goes
    single = [(1, "BIRLA INSTITUTE"), (1, "text"), (1, "Page 1 of 1")]
    assert [text for _, text in h.drop_page_furniture(single)] == ["BIRLA INSTITUTE", "text"]


def test_section_items_split_bullets_and_sentences():
    glyphs = [(1, " Strategy of process engineering design.  Use of process simulators for process creation.")]
    assert h.section_items(glyphs) == ["Strategy of process engineering design.", "Use of process simulators for process creation."]
    lettered = [(1, "A. Understand systems engineering principles and life cycles."), (1, "B. Apply systems thinking to infrastructure.")]
    assert h.section_items(lettered) == ["Understand systems engineering principles and life cycles.", "Apply systems thinking to infrastructure."]
    prose = [(1, "After completing this course, the student will be able to solve differential"), (1, "equations. understand whether a problem is solvable.")]
    assert h.section_items(prose) == ["Solve differential equations.", "Understand whether a problem is solvable."]
    inline = [(1, "A. To introduce advertising concepts b. To engage the students through activities c. To prepare them for practice")]
    assert h.section_items(inline) == ["To introduce advertising concepts", "To engage the students through activities", "To prepare them for practice"]
