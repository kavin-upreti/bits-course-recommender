# BITS Academic Course Recommender

A Django dashboard where a BITS Pilani student sets up a profile, sees what their degree still needs, plans a
clash-free timetable, and asks a chat assistant for electives ("Suggest DELs related to AI", "an OPEL with no
attendance requirement"). Every academic decision comes from the supplied BITS documents, processed into a database.
The LLM only reads the question and writes the answer.

How we got here, and why each piece is built the way it is: [`ideation_guide.md`](ideation_guide.md).

## The task

Postman Round 2 asks for an agentic course recommender that:

1. **Uses the supplied data as the source of truth**: Academic Regulations, the Bulletin, this semester's timetable,
   540 course handouts, and a student profile. No invented rules or course properties.
2. **Pre-processes the PDFs into a clean, structured dataset** (courses, programme rules, handout facts, timetable,
   source references), flagging anything that can't be extracted reliably instead of guessing.
3. **Lets a student create and update a profile**: campus, batch, single / dual degree, semester, completed and
   current courses, minor, interests.
4. **Works out requirements first, then recommends**: remaining CDC / DEL / HUEL / OPEL, prerequisites and rules
   decide what is *allowed*; interests and the question only decide the *order*.
5. **Answers natural-language questions** and explains why each course fits. If a handout doesn't state something,
   it says "couldn't verify".
6. **Brownie points: timetable intelligence**: clash-free sections, exam clashes, no 8 AM, a free weekday, compact days.

How each point is met:

| Asked | Where |
|---|---|
| Structured dataset from the PDFs | `extractors/` → `dataset/code processed/*.json`, hand-checked rules in `dataset/manually processed/`, loaded by `manage.py ingest` into `catalog/models.py`. Every record keeps `sources` (document, page) and `needs_verification`. |
| Profile | `/profile/`: the BITS ID gives batch, campus and degree(s); the programme chart fills in compulsory courses; electives, minor, interests and strengths are added by hand. |
| Requirements before recommendations | `recommender/requirements.py`, `recommender/eligible.py` stage A (offered, category, not done, prerequisites, batch rules). |
| Natural-language queries | `/chat/`: the LLM turns the message into tool calls; `recommender/eligible.py` does the rest. |
| Handout preferences, "couldn't verify" | `recommender/handout_facts.py`: tri-state facts (yes / no / unknown with a reason). |
| Timetable intelligence | `recommender/timetable.py` and `plan.py`: section search, lunch hour, exam and unit checks; the `/semester/` page and every recommendation use it. |

## Run it

Needs Python 3.11 or newer (built on 3.14). About 2 GB of disk for the local models.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env                      # then add at least one free LLM key (below)
.venv/bin/python manage.py migrate
.venv/bin/python manage.py ingest         # loads dataset/ into SQLite, builds embeddings (~20 s + a one-time model download)
.venv/bin/python manage.py runserver      # open http://127.0.0.1:8000, register with your BITS ID
```

`.env` keys (see `.env.example`):

| Key | What |
|---|---|
| `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY` | Free keys from console.groq.com, openrouter.ai, aistudio.google.com. One is enough; a provider without a key is skipped. |
| `LLM_PROVIDERS` | `provider:model` list, tried in order. A rate-limited provider falls through to the next. |
| `DJANGO_SECRET_KEY`, `DJANGO_DEBUG` | Django basics. `DJANGO_DEBUG=1` also shows a debug panel on the chat page. |

Everything is free: the embedding and reranking models run locally (Apple MPS or CPU), and the LLMs are free tiers.

The processed data is committed, so the steps above don't need the PDFs. To rebuild the data from scratch, put the
PDFs in `dataset/raw/` (`timetable.pdf`, `bulletin.pdf`, `Academic-Regulations-2023.pdf`, `handouts/*.pdf`) and run,
in order:

```bash
.venv/bin/python extractors/timetable.py
.venv/bin/python extractors/bulletin.py
.venv/bin/python extractors/build_handout_overrides.py
.venv/bin/python extractors/handouts.py   # ~1.5 min, two small local models
.venv/bin/python manage.py ingest
```

Tests: `.venv/bin/python manage.py test` (no network; the models and LLM are fakes) and `.venv/bin/python -m pytest tests/`.

## Using it

1. **Register** with your BITS ID (e.g. `2024A7PS0832P`, dual `2024B3A7PS0832P`) and a password.
2. **Profile**: the semester you're going into is worked out from your batch and the loaded timetable. Add a minor,
   interests, strengths, and optionally the courses you did well in or struggled with.
3. **Electives you've taken** (from 2-1 on). Compulsory courses are already filled in from your chart.
4. **Home**: credits done, what's left (DEL / HUEL / OPEL, minor), and your chart semester by semester.
5. **Semester**: every clash-free timetable for this semester's courses, with filters and an exam calendar.
6. **Assistant**: ask in plain words. Each suggestion is a card with its handout facts and why it matches. Tick
   courses to preview them in your timetable, then add them to your semester.

## How it works

```
message ─> LLM picks a tool + arguments ─> get_eligible_courses (Python, no LLM)
                                            A  rules: offered, category, not done, prerequisites, batch
                                            B  handout filters (no midsem, open book, ...) → unknown = "couldn't verify"
                                            C  ranking: embeddings + reranker, relevance cutoff, profile boost
                                            D  soft preferences (8 AM, free day, disliked evaluation styles)
                                            E  fit with the current courses (sections, exams, units)
        reply <─ LLM writes it from the results; cards and "not shown, and why" come from the tool results
```

The LLM never decides what a student can take, never sees the student's data, and can't name a course no tool
returned (a guardrail checks every code in the reply). Machine learning is used for topic matching (`e5-small-v2`
embeddings + `ms-marco-MiniLM` reranker), for finding the same class under two codes, and for reading attendance /
make-up wording in handouts. Details in the ideation guide.

## Evaluation

**Ranking, no LLM** (`manage.py embedding_eda` → `eda/embedding_eda.md`): 14 topic queries with hand-picked
correct courses, over the 539 offered courses, using the app's own code.

| | hit@5 | MRR | per query |
|---|---|---|---|
| **e5-small-v2 + ms-marco reranker (used)** | **0.80** | **0.82** | 0.2 s |
| best without a reranker (bge-small) | 0.79 | 0.84 | 0.01 s |
| e5-small-v2 + bge-reranker-base | 0.63 | 0.78 | 0.8 s |

The profile boost was checked on the same queries under 5 different student profiles: hit@5 never dropped.

**The whole assistant, real LLM** (`manage.py chat_eval` → `eda/chat_eval.md`): 30 messages from three
students, including the brief's four examples. Latest run: the right tool call in 28 of 30 (the other two read
"a HUEL" as count 1), 69 of 82 expected courses reached the cards, 0 invented courses, 0 tool errors, and all 3
topics nothing covers (cooking, marine biology) answered with "nothing matches".

## Limitations

- **The LLM is a free tier and the weakest link.** It can misread a category or leave a course out of its reply;
  the guards catch invented courses and missed categories, not every misreading.
- 113 of the 539 offered courses have no handout, so they're matched on the title and Bulletin text only.
- One message at a time: no chat history.
- The profile boost is content similarity, not a grade prediction. Strengths are searched like topics, so write them
  out ("machine learning", not "ML").
- The 9 branch cases in `chat_cases.json` ("maths courses on probability") weren't fully re-run after the last prompt
  change (free quotas ran out).
- Old-curriculum codes (e.g. MATH F111 Mathematics I) can be suggested to newer batches who did the new code
  (MATH F101): the data has no mapping saying they're the same course.
- 2+2 International Collaborative Programmes (two years at BITS, then RMIT, Iowa State, Buffalo or RPI) aren't
  supported; the brief names only single and dual degrees. The Bulletin (IV-142 to IV-223) gives them their own
  rules: a separate HUEL / DEL / OPEL split, capstones instead of PS-II or thesis, swapped first-year courses, and
  (for Iowa State) HUELs from a pool the Bulletin never lists. A 2+2 student would get a regular B.E. plan.
- A prerequisite met by a current course blocks the course this semester (the Regulations may allow concurrent
  registration).
- After a separate `ingest`, restart the server (caches are per process).

## Commands

All `.venv/bin/python manage.py ...`:

| Command | What |
|---|---|
| `ingest [--skip-embeddings] [--wipe-students]` | Reload the catalog from `dataset/`, then build pieces, embeddings and same-class pairs |
| `build_embeddings` | Rebuild pieces, embeddings and same-class pairs only |
| `embedding_eda`, `equivalence_eda` | Ranking and same-class evaluations → `eda/` |
| `chat_eval [--only N ...]` | The whole assistant with the real LLM → `eda/chat_eval.md` |
| `handout_facts_report` | How many courses have each handout fact unknown |
| `llm_smoke_test` | Say hello to each LLM provider (checks keys and model names) |

## Layout

```
extractors/        PDF -> JSON (timetable, Bulletin, handouts) + hand-checked handout values
dataset/           code processed/ (extractor output), manually processed/ (hand-curated rules)
bits_recommender/  the Django project: settings and URLs (the three apps below do the work)
catalog/           app: course / programme / rule models and the ingest command
students/          app: registration, profile, home, semester and course pages
recommender/       app: requirements, eligibility, ranking, timetable search, the LLM agent and chat page
eda/               evaluation reports (ranking, same-class detection, the chat assistant)
tests/             extractor rule tests (app tests live in each app's tests/)
```
