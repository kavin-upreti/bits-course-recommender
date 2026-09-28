# BITS Academic Course Recommender

A web app for BITS Pilani students. You make a profile, see what your degree still needs, plan a clash-free
timetable, and ask a chat assistant for electives, like "Suggest DELs related to AI" or "an OPEL with no attendance
requirement".

All the course information comes from the BITS documents we were given, turned into a database. The AI model only
reads your question and writes the answer. Everything in between (what you're allowed to take, what fits your
timetable) is worked out by normal code.

Want to know how and why it was built this way? Read [`ideation_guide.md`](ideation_guide.md).

## What the task asked for

Postman Round 2 asked for a course recommender that:

1. **Uses only the BITS documents** we were given (Academic Regulations, Bulletin, this semester's timetable, 540
   course handouts) and never makes up rules or course details.
2. **Turns those PDFs into clean, structured data first**, and marks anything it couldn't read reliably instead of
   guessing.
3. **Lets a student set up a profile**: campus, batch, single or dual degree, semester, courses done, minor, interests.
4. **Checks requirements before recommending**: what you still need (CDC, DEL, HUEL, OPEL) and what you're allowed
   to take decide the list; your interests only decide the order.
5. **Answers plain-English questions** and says why each course fits. If a handout doesn't say something, it says
   "couldn't verify" instead of guessing.
6. **Bonus: timetable help**: no clashes, no exam clashes, fewer 8 AM classes, a free day.

Where each one lives:

| Asked for | Where |
|---|---|
| Clean data from the PDFs | `extractors/` reads the PDFs into `dataset/`; `manage.py ingest` loads it into the database. Every record keeps where it came from (document and page). |
| Profile | The profile page. Your BITS ID gives your batch, campus and degree; your programme's chart fills in your compulsory courses. |
| Requirements first | `recommender/requirements.py` and the first stage of `recommender/eligible.py`. |
| Plain-English questions | The chat page. The AI picks a search; `recommender/eligible.py` does the searching. |
| "Couldn't verify" | `recommender/handout_facts.py`: every handout fact is yes, no, or unknown (with the reason). |
| Timetable help | `recommender/timetable.py` and `plan.py`, used by the Semester page and by every recommendation. |

## Run it on your computer

You need Python 3.12 or newer, internet for the first run, and about 2 GB of free space.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env                      # then paste in at least one free AI key (see below)
.venv/bin/python manage.py migrate
.venv/bin/python manage.py ingest         # loads the data and builds the search index (about 20 s)
.venv/bin/python manage.py runserver      # open http://127.0.0.1:8000 and register with your BITS ID
```

On Windows, use `py -m venv .venv`, then `.venv\Scripts\pip` and `.venv\Scripts\python` in the same commands, and
`copy` instead of `cp`.

`requirements.txt` has the exact versions this was built with, so you get the same results we do.

**AI keys** go in `.env`. They're free, and one is enough:

| Key | Where to get it |
|---|---|
| `GROQ_API_KEY` | console.groq.com |
| `OPENROUTER_API_KEY` | openrouter.ai |
| `GEMINI_API_KEY` | aistudio.google.com |

If one is busy or out of free uses, the next one answers. Leave a key empty to skip that provider. The search
models run on your own computer, so nothing costs money.

**Tests**: `.venv/bin/python manage.py test` (no internet needed) and `.venv/bin/python -m pytest tests/`.

**Rebuilding the data from the PDFs** (optional; the data is already in the repo): put the PDFs in `dataset/raw/`
(`timetable.pdf`, `bulletin.pdf`, `Academic-Regulations-2023.pdf`, `handouts/*.pdf`), then run:

```bash
.venv/bin/python extractors/timetable.py
.venv/bin/python extractors/bulletin.py
.venv/bin/python extractors/build_handout_overrides.py
.venv/bin/python extractors/handouts.py   # takes about 1.5 minutes
.venv/bin/python manage.py ingest
```

## Using it

1. **Register** with your BITS ID (like `2024A7PS0832P`, or `2024B3A7PS0832P` for a dual degree) and a password.
2. **Profile**: add your minor, interests and strengths. If grades matter to you, pick the courses you did well in.
3. **Electives you've taken** (from 2-1 on). Compulsory courses are already filled in from your chart.
4. **Home**: credits done, what's left, and your whole chart.
5. **Semester**: every clash-free timetable for this semester, with filters and an exam calendar.
6. **Assistant**: ask in plain words. Each suggestion comes as a card with its handout details and why it matches.
   Select the ones you like, preview them in your timetable, then add them.

## How it works

```
your question ─> AI picks a search ─> get_eligible_courses (normal code, no AI)
                                        1. rules: offered, right category, not done, prerequisites met
                                        2. handout wishes (no midsem, open book...); unknown = "couldn't verify"
                                        3. ranking: local search models + your profile
                                        4. soft wishes (fewer 8 AMs, a free day)
                                        5. must fit with your current courses
             answer <─ AI writes it from the results; the cards come straight from the database
```

The AI never decides what you can take, never sees your data, and can't mention a course the search didn't
return. If it tries, it's asked to fix its answer, and those sentences are removed if it doesn't.

The search uses two small models that run on your computer: one finds courses whose text is close to your topic
(`e5-small-v2`), and one checks those more carefully (`ms-marco-MiniLM`).

## How well it works

**Search alone** (`manage.py embedding_eda`, full report in `eda/embedding_eda.md`): 14 test questions where we
picked the right answers by hand.

| | Right answers in the top 5 | How high the first right answer is (1 = always first) | Time per question |
|---|---|---|---|
| **The models we use** | **80%** | **0.82** | 0.2 s |
| Without the checking model | 79% | 0.84 | 0.01 s |
| A bigger checking model | 63% | 0.78 | 0.8 s |

Adding your profile to the ranking never made these results worse (tested with 5 different student profiles).

**The whole assistant** (`manage.py chat_eval`, report in `eda/chat_eval.md`): 30 real questions from three students,
including the four examples from the task. The AI chose the right search 28 times out of 30, 69 of 82 expected
courses reached the cards, it never made up a course, and it correctly said "nothing matches" for topics no course
covers (like cooking).

## Known limits

- **The AI is a free version**, so it's the weakest part. It can sometimes misread a request or leave a course out
  of its answer.
- 113 of the 539 courses this semester have no handout, so they're matched on their title and Bulletin description
  only.
- Each question is answered on its own; it doesn't remember earlier messages.
- Your profile changes the order by how similar courses are, not by predicting grades. Write strengths out in full
  ("machine learning", not "ML").
- Questions limited to one subject ("maths courses on probability") weren't re-tested with the real AI after the
  last change, because the free AI limits ran out.
- Old course codes (like MATH F111 Mathematics I) can be suggested to newer batches who took the new version
  (MATH F101), because the data doesn't say they're the same course.
- **2+2 programmes** (two years at BITS, then RMIT, Iowa State, Buffalo or RPI) aren't supported, because the task
  only mentions single and dual degrees. The Bulletin gives them different rules: their own elective counts, capstone
  projects instead of PS-II or thesis, some swapped first-year courses, and (for Iowa State) HUELs from a list the
  Bulletin never prints. A 2+2 student would get a normal B.E. plan here.
- If a prerequisite is a course you're taking right now, the app waits until next semester (the rules might allow
  taking both together).
- After running `ingest` again, restart the server.

## Commands

All start with `.venv/bin/python manage.py`:

| Command | What it does |
|---|---|
| `ingest` | Loads all the data into the database and builds the search index |
| `build_embeddings` | Rebuilds just the search index |
| `embedding_eda`, `equivalence_eda` | Tests the search and the "same course, two codes" detection → `eda/` |
| `chat_eval` | Tests the whole assistant with the real AI → `eda/chat_eval.md` |
| `handout_facts_report` | Shows how many courses are missing each handout detail |
| `llm_smoke_test` | Checks your AI keys work |

## Folders

```
extractors/        reads the PDFs into JSON
dataset/           the data: code processed/ (from the PDFs), manually processed/ (typed in by hand)
bits_recommender/  Django's settings and URL list (the three folders below do the real work)
catalog/           the course data and the ingest command
students/          registration, profile, home, semester and course pages
recommender/       requirements, search, ranking, timetables, the AI assistant
eda/               test reports
tests/             tests for the PDF readers (each app also has its own tests)
```
