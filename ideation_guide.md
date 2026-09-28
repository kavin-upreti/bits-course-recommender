# Ideation guide

How we thought about the problem, what we tried, what we kept, and why. The [README](README.md) says how to run it.

## 1. Where we started

The first idea was simple: an AI reads the student's message, turns it into a list of wants and don't-wants, and a
scorer picks courses. But after reading the task again, we saw that the hard part isn't the chat. It's knowing what a
student is actually *allowed* to take. So we did things in this order: the data first, then the rules, and the AI last.

A few rules we stuck to the whole way:

- **Never guess.** If a document doesn't say something, we store "unknown", not "no". Students see
  "couldn't verify" with the reason.
- **The PDF readers know how the documents are laid out, not what's in them.** Nothing like course codes or dates is
  hardcoded, so next semester's PDFs go through the same code.
- **Everything remembers where it came from** (which document and page).
- **Anything that decides eligibility is normal code**, so it can be tested and explained.

## 2. Getting the data out of the PDFs

| Document | How we read it | What came out |
|---|---|---|
| Timetable | Table reading (pdfplumber); columns are found by their names | All 728 course entries: sections, rooms, times, exams, instructors. Checked by hand: exact. |
| Bulletin | Page text and tables, split by the Bulletin's own headings | 103 programme charts, 28 CDC lists, 23 minors, 136 HUELs, 30 audit courses, 2,006 course descriptions with prerequisites |
| Handouts (540) | Rules for the structure (headings, tables); two small models on our own computer for the attendance and make-up wording | About 425 read fully by code |
| Regulations | Read by hand into `dataset/manually processed/`, with the exact quote for each rule | Rules like the unit cap, lunch hour and higher-degree limit |

Some things had to be done by hand, and they're labelled as such:

- **113 handouts** had unusual layouts. Our first handout reader tried to handle every layout and grew to about
  1,500 lines. We replaced it with a simpler one (~550 lines) that handles the common layouts, and typed the rest in
  by hand (`extractors/build_handout_overrides.py`). Those records are marked `manually_checked`.
- **2 handouts were scanned images** with no text, so we typed them in and ran them through the same code.
- **The Bulletin contradicts itself in 3 CDC lists** (Environmental, ECE, Pharmacy). We fixed them by hand and
  stored the reason with each fix.
- **3 handout PDFs are actually another course's handout.** They stay flagged; fixing them needs the right PDFs.

A few judgement calls: "best 2 of 3 quizzes" stores each quiz at the weight it really counts for; handouts whose
weights don't add up to 100 are kept as printed, with a note; if a handout prints a different course code but the
same title, we treat it as the same course listed twice.

## 3. Working out where a student stands

- **The BITS ID tells us the degree.** `2024B3A7PS0832P` means 2024 batch, M.Sc. Economics + B.E. Computer Science,
  Pilani campus.
- **The semester being planned comes from the timetable** (2026-27 Sem 1 means the 2024 batch is going into 3-1).
- **The programme chart fills in compulsory courses.** Everything before that semester counts as done. When the
  chart says "X or Y", the student picks which one.
- **A course's type depends on your programme.** The same course can be a CDC for one degree and an OPEL for another,
  so it's worked out per student.
- **The totals come from real degree audits** (44 courses, 129 units), because the Bulletin we have is older than
  the current rules. The OPEL count is whatever's left. Dual degrees have no OPEL requirement at all.

## 4. Why not just ask an AI

Just giving an AI the profile and a course list ("what should I take?") doesn't meet the task:

- It **makes up courses and rules**. It can't know prerequisites, this semester's courses, or a student's chart.
- It **can't check handout details**. It will say "no midsem" confidently whether or not the handout says so.
- **The PDFs are too big** to send, and big chunks cost thousands of tokens per question on free plans.
- **The same question gets different answers**, so nothing can be tested.

So the AI does only two jobs: understand the question, and write the answer. It calls one of four tools
(`get_eligible_courses`, `check_plan`, `get_remaining_requirements`, `get_course_details`), and normal code does the
rest. Some safety checks around it:

- sloppy tool inputs are fixed (like `"3"` instead of `3`), and wrong ones are sent back to the AI to correct;
- if the student asked for a category the AI didn't search, it's reminded once;
- if the answer mentions a course no tool returned, the AI rewrites it, and those sentences are removed if it doesn't;
- the course cards and the "not shown, and why" list come from the database, never from the AI's text.

The AI never sees the student's data, and we only send what the answer needs. That brought one question down from
about 111,000 tokens (our first version) to 5,000–14,000. To keep it free, one small piece of code talks to Groq,
OpenRouter and Gemini, and moves to the next one if a service is busy.

## 5. Matching a topic to courses

- **Small pieces, not whole documents.** Each course is cut into short pieces (title, topics, outcomes, description
  sentences), about 15,500 in all. That way "transformers" matches the one line about transformers.
- **Find, then check.** One model quickly finds courses with similar text; a second model reads the question and each
  piece together, which is slower but much more accurate. We tried 3 finding models with and without 2 checking
  models on 14 test questions with hand-picked answers, and kept the best pair.
- **Only real matches.** A course needs a relevance of at least 0.75 to count. A short, honest list is better than
  five loose ones.
- **Each topic on its own.** "AI, ML, DL, NLP" mixed into one search half-matches everything. Each topic is searched
  separately and gets its own 30 candidates. (When topics shared 30, "programming, algorithms" lost Object Oriented
  Programming.)
- **When no handout uses the words.** Nothing this semester says "video editing", but the catalogue has a course named
  after it (not offered now). The offered courses closest to it are shown, clearly labelled "no direct match".
  A topic nothing covers (cooking) returns nothing.
- **Same course, two codes** (EEE F434 = ECE F434): if two courses share almost all their text, the student sees one
  card.

Things we tried and dropped: letting the AI expand topics into related ones (it drifted, and we couldn't explain it),
a hand-made topic list, and one search entry per whole handout.

## 6. Using the profile

The profile should change the order of results, never what's allowed, and never make normal searches worse.

| Profile field | What we did | Why |
|---|---|---|
| Interests, strengths | Split into short phrases ("programming and algorithms" becomes two), each searched like a topic; a real match gives a small boost | Comparing the whole text with a course's overall content missed obvious matches (strength "programming" vs OOP) |
| Did well in / struggled with (up to 5 each) | Courses similar to ones you did well in move up; similar to ones you struggled with, down a little | Easier than typing a grade for every course, and a clearer signal |
| "Grades matter a lot" | Doubles the "did well" part | For students who care about grades, similar courses are where they'll likely score well |
| A subject like "maths" | If you ask for one subject's courses ("maths courses on probability"), the search stays in that department. As a strength, it boosts that department | The search can't understand short forms like "maths" |
| SOP plan | A reminder to keep a slot free | Not a ranking thing |
| Weaknesses, job vs research goal, CGPA, grades per course | Dropped | "Rote learning" can't be read from a handout; we couldn't measure whether a course leans research or job; typing every grade was too much work |

Before keeping it, we checked the 14 test questions with 5 different student profiles. The right answers in the top
5 never went down.

One bug we caught late: at first, a word like "finance" anywhere in the question limited the search to finance
courses. But most subject words are also normal topics, so "courses on ML and finance" found only finance. Now the AI
says separately when a question is limited to one subject, and otherwise every word is just a topic.

## 7. Timetables

- **Trying section combinations one by one**, and backing out as soon as something clashes. Sections at exactly the
  same times count as one option (MATH F211 has 12 tutorials), and courses with fewer options go first.
- **Hard rules**: no two courses at once, a lunch hour free every day, no two exams at once, the unit cap, at most one
  higher-degree course.
- **Wishes** (fewer 8 AMs, a free day, fewer gaps, a teacher) only change the order, so there's always something to
  show, with a note when a wish can't be met.
- **Every recommended course is checked against your current courses** before it's shown. If nothing fits, it names
  the courses that clash and why.

## 8. How we tested it

- **189 automatic tests** that use fake models and a fake AI, so they never touch the internet.
- **Search tests** on 14 hand-labelled questions, and tests of the "same course, two codes" detection.
- **Whole-assistant tests** with the real AI: we compare its results with what the search returns for the right
  request, so we measure only the AI's part.
- **Small experiments before every tuned number**, written next to the number in `recommender/config.py`.

## 9. Ideas we didn't build

Remembering earlier messages ("swap the second one"), predicting grades, 2+2 international programmes (their rules are
in the Bulletin, but the task only mentions single and dual degrees), "courses at the same time as X", and ranking by
minor.
