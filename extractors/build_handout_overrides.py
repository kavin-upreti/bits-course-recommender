"""Builds dataset/manually processed/handout_overrides.json: values read by hand from handouts the extractor
can't parse (unusual layouts). handouts.py applies them after extraction and clears the matching issues.

Evaluation rows: (name, weight %, duration, date, remarks) exactly as printed in the PDF.
Run: .venv/bin/python extractors/build_handout_overrides.py
"""
import json

from handouts import MANUAL_PATH, component

OUTPUT = MANUAL_PATH.with_name("handout_overrides.json")

# ---- shared templates (the same printed scheme appears in several handouts)
PROJECT_A = [("Project Title & Plan of Work", 5, "", "2nd week", ""), ("Written Presentation-I", 5, "", "4th week", ""),
             ("Seminar-I / Viva-I", 10, "", "8th week", ""), ("Written Presentation II", 15, "", "8th week", ""),
             ("Seminar II / Viva II", 10, "", "12th week", ""), ("Final Report", 25, "", "16th week", ""),
             ("Weekly Interaction and Diary", 10, "", "16th week", ""), ("Final Seminar and Viva", 20, "", "16th week", "")]
PROJECT_B = [("Project Title & Plan of Work", None, "", "2nd week", "no weightage printed"), ("Viva I & Viva II", 10, "", "4th week", ""),
             ("Mid sem Seminar / Report", 10, "", "8th week", ""), ("Weekly report submission", 10, "", "every week", ""),
             ("Viva III / Viva IV", 10, "", "10th, 12th week", ""), ("Final Seminar", 15, "", "16th week", ""),
             ("Final Report", 15, "", "16th week", ""), ("Weekly Interaction and Diary", 10, "", "8th, 16th week", ""),
             ("Final Seminar and Viva", 20, "", "16th week", "")]
PROJECT_C = [("Project Title and Plan of Work", 5, "", "2nd Week", ""), ("Viva I and II", 10, "", "4th Week", ""),
             ("MidSem Seminar / Report", 20, "", "8th Week", "printed as 10+10"), ("Viva III and IV", 10, "", "12th Week", ""),
             ("Final Report", 25, "", "16th Week", ""), ("Weekly Interaction and Diary", 10, "", "16th Week", ""),
             ("Seminar and Viva", 20, "", "16th Week", "")]
LAB_PROJECT = [("Project Outline & Plan of Work", 5, "", "2nd week", ""), ("Literature Survey", 10, "", "4th week", ""),
               ("Lab Related Activities – 1", 25, "", "6th week", ""), ("Midsem Report", 5, "", "8th week", ""),
               ("Midsem Seminar", 5, "", "8th week", ""), ("Lab Related Activities – 2", 30, "", "12th week", ""),
               ("Final Report", 10, "", "16th week", ""), ("Final Seminar and Viva", 10, "", "16th week", "")]
THESIS = [("Viva-I", 15, "", "5th week", ""), ("Mid-sem written report", 15, "", "10th week", ""),
          ("Mid-sem presentation", 15, "", "10th week", ""), ("Viva-II", 15, "", "15th week", ""),
          ("Final Dissertation", 25, "", "Last day of class work", "evaluated jointly with the DRC examiner"),
          ("Final viva-voce", 15, "", "Actual date announced by DRC", "evaluated jointly with the DRC examiner")]
VENTURE = [("Mentor-Mentee Engagement", 10, "", "", ""), ("Class Participation", 10, "", "", ""),
           ("Key Deliverables (customer discovery, market validation, go-to-market)", 30, "", "", ""),
           ("Investor Documents and Pitch Presentations", 50, "", "", "evaluated by external jury members")]
BIO_G524 = [("Mid Semester test", 20, "90 mins", "", "CB/OB"), ("Surprise and announced Quizzes", 10, "15 min", "", "CB"),
            ("Experiments and Lab quiz", 30, "", "", "CB & OB"), ("Seminars/ Assignment", 10, "", "", "OB"),
            ("Comprehensive", 30, "2 hrs", "", "CB & OB")]
ECON_QUIZZES = [("Assignment & Class Participation", 5, "15 minutes", "See Weekly Workflow", "Open Book; group presentation"),
                ("Quiz – 1", 15, "45 minutes", "8th Sept 2026", "Closed Book"), ("Mid-Semester Exam", 25, "90 minutes", "8 Oct 2026 (AN1)", "Open Book"),
                ("Quiz – 2", 15, "45 minutes", "5th Nov 2026", "Closed Book"), ("Quiz – 3", 15, "45 minutes", "26th Nov 2026", "Closed Book"),
                ("Comprehensive Exam", 40, "180 minutes", "12 Dec 2026 (FN)", "Closed Book")]
EDP = [("Attendance-associated discussion - Workshop", 6, "", "Workshop", "18 marks"),
       ("Lab Assignments (Evaluatives)", 23.33, "", "As announced by IC", "70 marks; best 4 out of 6; no makeup"),
       ("Project", 17.33, "", "Two design review meetings", "52 marks"),
       ("Mid-semester test (On prototyping)", 20, "", "As announced by AUGSD", "60 marks"),
       ("Comprehensive exam (written + computer-based test)", 33.33, "", "As announced by AUGSD", "100 marks")]
BUSINESS_PLAN = [("Mid Semester Exam", 30, "90 minutes", "08/10 AN1", "Close Book"), ("Surprise Quizzes", 10, "", "", "Close Book; 1 quiz in buffer"),
                 ("Assignments/Exercises", 15, "", "To be announced in the class", "1 assignment in buffer"),
                 ("Business Plan Preparation and Presentation", 15, "", "To be announced in the class", ""),
                 ("Comprehensive Exam", 30, "180 minutes", "12/12 FN", "Partially Close and Open Book")]
SCM_G515 = [("Mid-Semester Examination", 25, "90 min", "8/10 2:00-3:30 PM", "Closed book"), ("Article Presentations", 15, "", "Continuous", "Open book"),
            ("Case Study", 15, "", "Continuous", "Open book"), ("Class surprise quizzes (n-2)", 10, "10 min", "In class hour", "Open book"),
            ("Comprehensive Examination", 35, "180 min", "10/12 FN", "Closed book")]
MPBA_G504 = [("Class discussions (linked to attendance, class assignment/quiz)", 5, "", "Every class", "Open Book"),
             ("Project (participation + report)", 15, "", "TBA", "Open Book"), ("Case Study/Simulation", 20, "", "Continuous", "Open Book"),
             ("Mid Semester Test", 25, "90 min", "05/10/2026 FN2", "Close Book"), ("Comprehensive Exam", 35, "180 min", "11/12/2026 FN", "Close Book")]
MPBA_G506 = [("Pre mid-sem Quizzes", 10, "", "TBA in Class", "Open Book"), ("Assignments", 10, "", "TBA in Class", "Open Book"),
             ("Pre-Mid Lab tests", 10, "", "TBA in Class", "Open Book"), ("Mid-term exam", 20, "90 min", "9th October 2026 4:00-5:30 PM", "Closed book"),
             ("In-class/lab Post-Mid Assignment", 10, "", "Weekly once", "Open book/notes/Internet"),
             ("Post-MID Sem Quiz", 10, "", "1st Week of November", "TBA"), ("Comprehensive Exam", 30, "3 hours", "14th December 2026 2-5 PM", "Closed book")]
MPBA_G507 = [("R module: In-class/lab Assignment", 5, "", "Weekly once", "Open book; 10% of the R module"),
             ("R module: Group Project", 5, "", "28th Sept", "Open book; 10% of the R module"),
             ("R module: Pre-MID Sem Quiz", 5, "30 mins", "3rd Week of September", "Closed book; 10% of the R module"),
             ("R module: Mid Semester", 10, "90 min", "2nd Week of October", "Closed book; 20% of the R module"),
             ("Python module: Comprehensive exam", 22.5, "180 min", "TBD", "45% of the Python module"),
             ("Python module: Quiz (Post-Mid)", 7.5, "15 min", "2nd week of Nov", "15% of the Python module"),
             ("Python module: In-Class Assignments & Group Project", 20, "", "TBD", "40% of the Python module")]
SMART_MATERIALS = [("Quizzes 1, 2 and 3", 10, "20 min", "Two announced, one unannounced", "Closed book; best two count, 5% each"),
                   ("Mid-Semester Examination", 30, "90 min", "As per the timetable", "Closed book"),
                   ("Demonstrations, Experiments, Project and Assignments", 20, "", "Continuous", "Open book"),
                   ("Comprehensive Examination", 40, "3 hours", "As per the timetable", "Closed book/open book as announced")]
CHEM_G553 = [("Mid Semester Test", 30, "", "06/10 FN2", ""), ("Assignments + Seminars + Term Papers", 30, "", "Continuous", ""),
             ("Comprehensive Examination", 40, "", "05/12 AN", "")]
EEE_G516 = [("Quiz", 10, "40 minutes", "TBA", "Closed book"), ("Mid-Semester Test", 20, "90 minutes", "Oct. 6th (FN2)", "CB"),
            ("Labs & Class tasks", 15, "", "TBA", "Open book; printed as 10% + 5%"), ("Team project", 15, "", "TBA", "OB; seminar 5%, work done and report 10%"),
            ("Comprehensive Examination", 40, "3 hrs", "Dec. 5th (AN)", "OB/OB+CB")]
MARKS_200 = [("Attendance & Classroom Interaction", 5, "", "All Lectures", "10 of 200 marks"),
             ("Lab component/ Opinionnaires/ Assignment/ Case study", 10, "", "Will be announced", "10x2 = 20 of 200 marks"),
             ("Quiz 1", 15, "50 min", "September (2nd Week)", "CB; 30 of 200 marks"), ("Quiz 2", 15, "50 min", "November (2nd Week)", "CB; 30 of 200 marks"),
             ("Mid Sem Exam", 25, "90 min", "", "CB; 50 of 200 marks"), ("Comprehensive Examination", 30, "3 hours", "", "OB; 60 of 200 marks")]

EVALUATIONS = {
    "018_BIO_G524.pdf": BIO_G524, "019_BIO_G524.pdf": BIO_G524,
    "029_BITS_E574.pdf": [("Seminar I", 20, "", "To be announced", ""), ("Seminar II", 20, "", "To be announced", ""),
                          ("Seminar III", 25, "", "To be announced by the DCA", ""), ("Final Project Report", 35, "", "", "")],
    "033_BITS_F103.pdf": EDP, "069_BITS_U103.pdf": EDP,
    "039_BITS_F219.pdf": [("Continuous Evaluation", 10, "30 min", "TBA", "Best 3 of 4, Closed book"), ("Lab component", 20, "", "In lab", ""),
                          ("Mid Semester", 30, "90 min", "As announced in the timetable", "Open/Closed book"),
                          ("Comprehensive", 40, "180 min", "As announced in the timetable", "Closed book")],
    "043_BITS_F314.pdf": [("Quiz – 1", 7.5, "45 minutes", "23 Sept 2026", "Closed Book; best two of three quizzes count (30% together)"),
                          ("Mid-Semester Examination", 30, "90 minutes", "06 Oct 2026 (AN1)", "Open Book"),
                          ("Quiz – 2", 7.5, "45 minutes", "06 Nov 2026", "Closed Book; best two of three"),
                          ("Quiz – 3", 7.5, "45 minutes", "23 Nov 2026", "Closed Book; best two of three"),
                          ("Comprehensive Exam", 40, "180 minutes", "07 Dec 2026 (FN)", "Closed Book")],
    "053_BITS_F437.pdf": [("Midsemester Test", 30, "90 mins", "", "Closed Book"),
                          ("Assignments & Classroom Participation", 30, "", "", "Open Book; printed as 20+10"),
                          ("Comprehensive Examination", 40, "180 mins", "", "Closed Book")],
    "056_BITS_F463.pdf": [("Quizzes Q1-Q4 (best 2 of 4)", 20, "0.5 hour each", "TBA", "Closed Book; 10% each, best two count"),
                          ("Programming Assignment", 10, "", "TBA", ""),
                          ("Mid-Semester Examination", 30, "1.5 hours", "07-10-2026, FN1", "Closed Book"),
                          ("Comprehensive Examination", 40, "3 hours", "08-12-2026, FN", "Open Book")],
    "059_BITS_F468.pdf": VENTURE, "229_ECON_F415.pdf": VENTURE,
    "061_BITS_F482.pdf": BUSINESS_PLAN, "228_ECON_F414.pdf": BUSINESS_PLAN,
    "063_BITS_G511.pdf": [("Research Plan", 10, "", "", "Format, objectives, literature survey and presentation"),
                          ("Development related activities", 20, "", "", "Report (5+5), viva/presentation (5+5)"),
                          ("Research related activities", 70, "", "", "Day-to-day observation 20, reports 30, seminar/viva 20")],
    "064_BITS_G540.pdf": [("Research Proposal", 25, "", "", "Format/contents 5, presentation 20"),
                          ("Development related activities", 25, "", "", "Report 15, viva/presentation 10"),
                          ("Research related activities", 50, "", "", "Reports 30, seminar/presentation 20")],
    "065_BITS_G562T.pdf": THESIS, "066_BITS_G563T.pdf": THESIS, "067_BITS_G564T.pdf": THESIS,
    "075_CE_F211.pdf": [("Tutorials (best 4 of 5)", 20, "50 minutes each", "Aug 17, Sep 7, Sep 21, Oct 26, Nov 16 (Tut hour)", "Open book; 5% each"),
                        ("Mid-Semester Examination", 35, "1.5 hours", "", "Closed book"), ("Comprehensive Examination", 45, "3 hours", "", "Closed book")],
    "076_CE_F213.pdf": [("Mid-semester Exam", 25, "90 minutes", "As per the timetable", "Closed book"),
                        ("Comprehensive Exam", 35, "3 hours", "As per the timetable", "Closed Book"),
                        ("Tutorials (Best 3 out of 5)", 15, "50 minutes", "Will be notified", "Open/Closed Book"),
                        ("Lab Quiz", 5, "", "Will be notified", "Open/Closed Book"),
                        ("Lab component with Lab Record (handwritten)", 10, "", "Each week", "Open Book"),
                        ("Class Quizzes", 10, "", "Every class", "Open/Closed Book")],
    "115_CE_G619.pdf": [("Mid-Semester Test", 20, "1.5 hours", "As per timetable", "CB"),
                        ("Project, Assignments, Computer Programs, Seminars, Take Home Tests, Class Test", 35, "", "Continuous", "OB"),
                        ("Comprehensive Examination", 45, "3 hours", "As per timetable", "CB")],
    "123_CHE_F312.pdf": [("Experiment Reports (data analysis & visualization, critical thinking, discussion)", 40, "", "Regular Laboratory Hours", "Open Book; 120 marks"),
                         ("Cycle 1 (or 2) Lab Exam", 15, "120 min", "29.09.2026 and 30.09.2026", "Closed Book"),
                         ("Cycle 2 (or 1) Lab Exam", 15, "120 min", "23.11.2026 and 25.11.2026", "Closed Book"),
                         ("Comprehensive Quiz", 15, "60 min", "20.11.2026", "Closed Book"),
                         ("Comprehensive Viva-Voce Exam", 15, "10 min", "26.11.2026 and 27.11.2026", "Closed Book")],
    "143_CHEM_F242.pdf": [("Laboratory Work & Reports", 84, "", "Continuous", "252 marks"),
                          ("Lab tests I and II", 16, "", "To be announced", "48 marks")],
    "156_CHEM_G553.pdf": CHEM_G553, "157_CHEM_G553.pdf": CHEM_G553,
    "164_CS_F301.pdf": [("Quiz 1", 12.5, "35–40 mins", "2nd September 2026", "Closed Book"),
                        ("Quiz 2", 12.5, "35–40 mins", "30th September 2026", "Closed Book"),
                        ("Mid-Semester Test", 30, "90 mins", "8th October 2026 (FN)", "Open Book"),
                        ("Comprehensive Examination", 45, "180 mins", "1st December 2026 (FN)", "Closed Book")],
    "176_CS_G525.pdf": [("Research Paper Debate", 10, "In class", "Throughout semester", ""),
                        ("Research Project (proposal 5, progress reviews 5+5, final work & presentation 15+5)", 35, "", "Throughout semester", "group"),
                        ("Mid-Semester Examination", 20, "90 min", "09/10 FN2", "Individual; Closed Book"),
                        ("Comprehensive Examination", 35, "180 min", "13/12 AN", "Individual; Closed Book")],
    "208_ECE_F314.pdf": [("Mid-sem Test", 30, "90 min", "", "Closed Book"), ("Quizzes", 30, "", "TBA", "Closed Book"),
                         ("Comprehensive Exam", 40, "180 min", "", "Closed/Open Book")],
    "218_ECON_F213.pdf": [(n, 7.5 if n.startswith("Quiz") else w, d, t, r + ("; best two of three quizzes count" if n.startswith("Quiz") else ""))
                          for n, w, d, t, r in ECON_QUIZZES],
    "219_ECON_F214.pdf": [("Quiz-1", 5, "30 Min", "21st August, 2026", "CB; best two of three quizzes count"),
                          ("Quiz-2", 5, "30 Min", "18th September, 2026", "CB; best two of three"),
                          ("Mid-Semester Examination", 30, "90 Min", "See the timetable", "CB"),
                          ("Quiz-3", 5, "30 Min", "6th November, 2026", "CB; best two of three"),
                          ("Comprehensive Examination", 50, "180 Min", "See the timetable", "CB/OB")],
    "222_ECON_F313.pdf": [("Mid-Semester Test", 30, "90 minutes", "06-10-2026", "CB"), ("Comprehensive Exam", 40, "180 minutes", "05-12-2026", "CB/OB"),
                          ("Class participation/attendance; Homework/ Assignments", 10, "", "23-11-2026", "OB"),
                          ("Tut Test 1", 20 / 3, "30 minutes", "24-09-2026", "CB; best two of three tut tests count (20%)"),
                          ("Tut Test 2", 20 / 3, "30 minutes", "12-11-2026", "CB; best two of three"),
                          ("Tut Test 3", 20 / 3, "30 minutes", "19-11-2026", "CB; best two of three")],
    "265_EEE_G516.pdf": EEE_G516, "266_EEE_G516.pdf": EEE_G516,
    "312_GS_F366.pdf": LAB_PROJECT, "313_GS_F367.pdf": LAB_PROJECT,
    "314_GS_F376.pdf": PROJECT_C, "315_GS_F377.pdf": PROJECT_C,
    "324_HSS_F323.pdf": MARKS_200, "326_HSS_F328.pdf": MARKS_200,
    "327_HSS_F329.pdf": [("Mid Sem", 25, "90 min", "10/10/2026 AN1", "OB"), ("Practical Test", 20, "50 min", "To be announced", ""),
                         ("Quiz", 10, "15 min", "To be announced", ""), ("Home Assignment", 10, "", "To be announced", ""),
                         ("Comprehensive", 35, "3 hours", "15/12/2026 FN", "")],
    "329_HSS_F343.pdf": [("Assignment", 15, "", "To be announced", "Open Book"), ("Mid Semester Exam", 30, "90 Minutes", "", "Close Book"),
                         ("Case Study", 15, "50 Minutes", "To be announced", "Open Book"), ("Comprehensive Examination", 40, "3 Hours", "", "Close Book")],
    "351_MAC_F312.pdf": [("Quizzes Q1-Q3 (best 2 of 3)", 10, "15 mins each", "TBA", "Closed Book; 5% each"),
                         ("Lab Assignment", 10, "Throughout Semester", "TBA", ""), ("Self Study Report", 20, "", "October 28, 2026", ""),
                         ("Mid-Semester Exam", 25, "1.5 Hours", "10 October FN1", "Closed Book"), ("Comprehensive Examination", 35, "3 Hours", "15 December FN", "Closed Book")],
    "383_ME_F316.pdf": [("Mid-Semester Test", 30, "90 Min.", "8/10 FN1", "Closed Book"), ("Comprehensive Examination", 40, "180 Min.", "1/12 FN", "Closed Book"),
                        ("Tutorials and Quiz", 30, "", "Tutorial Class", "Open Book; printed as 20 + 10; best four tutorial tests count")],
    "391_ME_F428.pdf": SMART_MATERIALS, "409_ME_U428.pdf": SMART_MATERIALS,
    "431_MF_F418.pdf": [("Mid Semester Test", 30, "90 minutes", "10.10.2026 AN1", "Closed Book"),
                        ("Quizzes (Surprise) (best n-2)", 10, "15 minutes", "Lecture session", "Closed Book"),
                        ("Case Study/group Assignments/Project", 20, "", "", "Open Book"),
                        ("Comprehensive Examination", 40, "3 hours", "16.12.2026 FN", "Closed Book/Open Book")],
    "439_MPBA_G504.pdf": MPBA_G504, "440_MPBA_G504.pdf": MPBA_G504,
    "443_MPBA_G506.pdf": MPBA_G506, "444_MPBA_G506.pdf": MPBA_G506,
    "445_MPBA_G507.pdf": MPBA_G507, "446_MPBA_G507.pdf": MPBA_G507,
    "451_MPBA_G517.pdf": [("Lab Evaluation", 5, "", "", ""), ("Class participation", 5, "", "", ""),
                          ("Mid-semester evaluation", 25, "1.5 h", "09/10 AN2", ""), ("Mid-semester Group Project", 15, "", "", ""),
                          ("End-semester Group Project", 15, "", "", ""), ("Comprehensive Examination", 35, "3 h", "14/12 AN", "")],
    "513_PHY_F312.pdf": [("Mid-Semester Examination", 30, "90 mins", "TBA", "Open/Closed; 60 marks"),
                         ("Comprehensive Examination", 40, "180 mins", "TBA", "Open/Closed; 80 marks"),
                         ("Class Tests/Assignments (best 3 of 4)", 30, "40 min each", "TBA", "Open/Closed; 60 marks")],
    "531_SCM_G515.pdf": SCM_G515, "532_SCM_G515.pdf": SCM_G515,
}
for name in ("206_ECE_F266", "215_ECE_F491", "240_EEE_F266", "256_EEE_F491", "303_GS_F266", "316_GS_F491", "322_HSS_F266", "336_INSTR_F266"):
    EVALUATIONS[f"{name}.pdf"] = PROJECT_A
for name in ("210_ECE_F366", "211_ECE_F367", "212_ECE_F376", "213_ECE_F377", "245_EEE_F366", "246_EEE_F367", "247_EEE_F376",
             "248_EEE_F377", "340_INSTR_F366", "341_INSTR_F367", "342_INSTR_F376", "343_INSTR_F377", "344_INSTR_F491"):
    EVALUATIONS[f"{name}.pdf"] = PROJECT_B
EVALUATIONS["177_CS_G525.pdf"] = EVALUATIONS["176_CS_G525.pdf"]

# The handout's own printed weights don't add up to 100; kept as printed, recorded as a note.
PRINTED_SUM = {
    "137_CHE_G554.pdf": ("95", [("Mid-Semester Test", 25, "90 Min.", "", "CB"), ("Final Project", 15, "2 to 3 weeks", "To be announced in the class", "Take home"),
                                ("Case Studies/Simulations/Assignments/lecture tests (4)", 20, "continuous", "", "Take Home"),
                                ("Comprehensive Examination", 35, "180 Min.", "", "CB & OB")]),
    "301_GS_F243.pdf": ("95 + up to 5% class participation", [("Homework Assignment", 10, "", "By 15th November 2026", "Open Book"),
                                                              ("Mid Semester Test", 30, "90 Minutes", "7/10 AN1", "Close Book"),
                                                              ("Class Assignments", 15, "", "Second/Third week of September 2026", "Open Book"),
                                                              ("Comprehensive Exam", 40, "180 Minutes", "10/12 FN", "Close Book"),
                                                              ("Class participation", 5, "", "", "up to 5%, from the note under the table")]),
    "318_HSS_F222.pdf": ("95 + up to 5% class participation", [("Homework Assignment", 10, "", "By 15 November 2026", "Open Book"),
                                                              ("Mid Semester Test", 30, "", "6/10 FN1", "Close Book"),
                                                              ("Class Assignments", 15, "", "Second/third week of September 2026", "Open Book"),
                                                              ("Comprehensive Exam", 40, "", "7/12 FN", "Close Book"),
                                                              ("Class participation", 5, "", "", "up to 5%, from the note under the table")]),
    "309_GS_F331.pdf": ("105", [("Assignment 1", 15, "", "TBA", ""), ("Mid-Semester Test", 30, "90 mins", "As announced", "Closed book"),
                                ("Assignment 2", 15, "", "TBA", ""), ("Comprehensive", 45, "180 min", "As announced", "Closed book")]),
    "332_HSS_U141.pdf": ("90 in the table + 10% class participation in the text", [
        ("Portfolio Evaluative 1", 15, "", "CCE", "Closed & open book"), ("Portfolio Evaluative 2", 20, "", "CCE", "Closed & open book"),
        ("Portfolio Evaluative 3", 25, "", "CCE", "Closed & open book"), ("Portfolio Evaluative 4", 30, "", "CCE", "Closed & open book"),
        ("Class participation and laboratory engagement", 10, "", "", "from the text above the table")]),
    "433_MGTS_U102.pdf": ("unclear (best-of rules on tutorials)", [
        ("Tut 1 (attendance)", 5, "50 min", "", "for attending"), ("Tut 2-4: Posters 1-3 (best 3 of 4 posters)", 60, "50 min each", "", "OB; 20% each"),
        ("Mid-Semester Test", 30, "40 min", "14/03", "CB"), ("Tut 5: Brainstorming session (attendance)", 5, "50 min", "", "for attending"),
        ("Tut 6-8: Group presentations", 20, "50 min", "", "plus 5% for attending 2 of the others")]),
    "434_MPBA_G501.pdf": ("90", [("Mid semester test", 25, "2 hrs", "10/10", "Closed book"), ("Case studies (2/3)", 10, "50 min", "TBA", "Open book"),
                                 ("Excel based exercise (1)", 5, "50 min", "TBA", "Open book"), ("Announced Quiz", 7, "30 min", "TBA", "Closed book"),
                                 ("Project: poster & viva", 8, "Take home", "TBA", "Open book; viva CB"),
                                 ("Comprehensive Exam", 35, "3 h", "16/12", "Partly Closed book")]),
    "452_MPBA_G520.pdf": ("105", [("Case study participation", 15, "", "", "Offline"), ("Quiz", 5, "", "", "Offline"), ("Course Project", 15, "", "", ""),
                                  ("Class participation", 10, "", "", ""), ("Mid-Semester Test", 25, "1.5 hours", "", "Closed Book"),
                                  ("Comprehensive Exam", 35, "3 hours", "", "Open Book")]),
}
PRINTED_SUM["453_MPBA_G520.pdf"] = PRINTED_SUM["452_MPBA_G520.pdf"]
PASS_FAIL = ["032_BITS_F101-1.pdf", "540_SW_E101.pdf"]  # Social Conduct: pass/fail, no weights


def rows(scheme):
    return [component(name, None if weight is None else round(weight, 2), duration, date, remarks) for name, weight, duration, date, remarks in scheme]


def build() -> dict:
    overrides = {name: {"evaluation": rows(scheme)} for name, scheme in EVALUATIONS.items()}
    for name, (total, scheme) in PRINTED_SUM.items():
        overrides[name] = {"evaluation": rows(scheme), "notes": [f"handout_weights_add_up_to: {total}"]}
    for name in PASS_FAIL:
        overrides[name] = {"evaluation": [], "notes": ["pass_fail_course"]}
    for name, fields in {**TOPICS_ABOUT}.items():
        overrides.setdefault(name, {}).update(fields)
    return overrides


from handout_overrides_text import TOPICS_ABOUT  # noqa: E402  (topics / about / titles read by hand)

if __name__ == "__main__":
    data = build()
    OUTPUT.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(f"{len(data)} handouts -> {OUTPUT}")
