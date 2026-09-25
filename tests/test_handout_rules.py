"""Checks for the rule-based parsing in extractors/handouts.py (no PDFs or models needed).

Run: .venv/bin/python -m pytest tests/
Each case is a real layout from a handout that was once parsed wrong.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extractors"))

import handouts as h  # noqa: E402


def test_parse_weight_forms():
    assert h.parse_weight("25%") == 25
    assert h.parse_weight("50 (25 %)") == 25        # a percent beats marks
    assert h.parse_weight("1 5 %") == 15            # letter-spaced
    assert h.parse_weight("20+10") == 30            # component in parts
    assert h.parse_weight("10% + 5%") == 15
    assert h.parse_weight("10x2=20 Marks") == 20
    assert h.parse_weight("60 /30", marks_then_percent=True) == 30
    assert h.parse_weight("10 (20 Marks)") == 10
    assert h.parse_weight("–") is None


def test_text_evaluation_rows():
    rows = ["Lab Related Activities – 1 25 6th week", "Mid-semester evaluation 1.5 h 25 09/10 AN2 TBA",
            "Tutorial 1 5% 50 minutes August 17th", "Mid Semester Test 30 06/10 FN2"]
    parsed = [(c.name, c.weightage_percent) for c in h.parse_evaluation_text(rows)]
    assert parsed == [("Lab Related Activities – 1", 25), ("Mid-semester evaluation", 25),
                      ("Tutorial 1", 5), ("Mid Semester Test", 30)]


def test_headings():
    assert h.match_heading("1: Course Description:")[0] == "description"
    assert h.match_heading("4.1 Textbook:")[0] == "textbooks"
    assert h.match_heading("Text & Reference Books: T1. Foundations")[0] == "textbooks"
    assert h.match_heading("AIMS AND LEARNING OBJECTIVE:")[0] == "objectives"
    # table rows and sentence fragments are not headings
    assert h.match_heading("Seminar III 25 To be announced by the DCA in consultation with") is None
    assert h.match_heading("course description.") is None
    assert h.match_heading("1. Project Outline & Plan of Work 5 2nd week") is None


def test_instructor_names():
    assert h.split_names("[Tufan Chandra Bera]") == [("Tufan Chandra Bera", False)]
    assert h.split_names("Dr. Bharat Richhariya [IC] [b.r@pilani.bits-pilani.ac.in]") == [("Dr. Bharat Richhariya", True)]
    assert h.split_names("Lecture: Amit Dua, and Mukesh Kumar Rohil (rohil@x.in)") == [("Amit Dua", False), ("Mukesh Kumar Rohil", False)]


def test_topic_prefixes():
    assert h.split_topic_cell("L.4. DNA replication L.5-L.6. DNA Transcription") == ["DNA replication", "DNA Transcription"]
    assert h.split_topic_cell("Lectures 1 to 3 – Intro") == ["Intro"]
    assert h.split_topic_cell("Total") == []


def test_component_patterns_need_word_start():
    # "tut" inside "institute" once made a make-up rule about exams decide the tutorial tests
    assert h.component_kind("Evaluative Tutorials") == "tutorial_test"
    assert not h.names_a_component("decision will be made as per institute norms")


def test_title_match_for_printed_code():
    assert h.title_matches("Introduction to Biological Sciences", "INTRO TO BIO SCIENCES")
    assert not h.title_matches("Microbiology", "ADV & APPLIED MICROBIO")
