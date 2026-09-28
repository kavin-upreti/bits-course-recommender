# BITS elective planner and course assistant

A Django app for BITS Pilani students: it works out what a student has done and still needs (CDCs, HUEL / DEL /
OPEL electives, minors), builds clash-free timetables for the semester, and has a chat assistant that recommends
electives ("Suggest DELs related to AI", "an OPEL with no attendance requirement") and can add them to the plan.

All course data comes from three PDFs (the timetable, the Bulletin, 540 course handouts), turned into JSON by the
extractors in `extractors/` and loaded into SQLite by `manage.py ingest`.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # then fill in the keys below
.venv/bin/python manage.py migrate
.venv/bin/python manage.py ingest          # loads the catalog, builds course pieces + embeddings (~20 s)
.venv/bin/python manage.py runserver       # register, fill the profile, then "Ask the course assistant"
```

| `.env` | What |
|---|---|
| `DJANGO_SECRET_KEY`, `DJANGO_DEBUG` | Django basics (`DJANGO_DEBUG=1` also shows the chat page's Debug panel) |
| `LLM_PROVIDERS` | `provider:model` list tried in order; a rate-limited or failing provider falls through to the next. Default: `groq:openai/gpt-oss-120b,openrouter:nvidia/nemotron-3-super-120b-a12b:free,gemini:gemini-3.5-flash` |
| `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY` | free keys (console.groq.com, openrouter.ai, aistudio.google.com); a provider without a key is skipped |
| `BITS_DATA_DIR` | optional; defaults to `./dataset` |

Everything is free: the embedding and reranking models run locally (Apple MPS or CPU), and the LLMs are free tiers.

Tests: `.venv/bin/python manage.py test` (no network: sockets are blocked, the embedder, reranker and LLM are
fakes, see `recommender/testing.py`) and `.venv/bin/python -m pytest tests/` (extractor rules).

## How the course assistant works

The LLM never decides what a student can take. It reads the message, calls tools with arguments, and writes the
reply from what the tools return. Every academic decision is Python and local models:

```
message ──> LLM picks tools + arguments ──> get_eligible_courses / check_plan / ... (Python, no LLM)
                                                │
          reply <── LLM writes it from the results; cards and the "not shown, and why" list come from the
                    database and the tool results, never from the LLM's text
```

**get_eligible_courses** (`recommender/eligible.py`), one stage per function:

1. **Rules**: offered this semester, in the asked category (a DEL counts as an OPEL once DELs are complete),
   not done (including equivalent codes), prerequisites met, batch rules, the student's exclusions.
2. **Handout filters** the student asked for (no midsem, at most N quizzes, open book, …), from facts parsed out
   of the handouts. Unknown facts don't fail a course; it goes to "couldn't verify" with the reason.
3. **Ranking** (ML, below).
4. **Soft preferences**: courses whose every section has an 8 AM class (or meets on the day to keep free), or with
   evaluation styles the student dislikes, stay in, a little lower (−0.05 each), with a note.
5. **Fit**: every returned course is checked against the student's current courses with the same timetable
   search the timetable page uses (classes, the lunch hour, midsem / compre dates, the unit limit, the
   higher-degree limit). A course that can't fit is listed under the cards with the exact reason.

**check_plan** checks several new courses together; the chat page runs it again when a course is ticked and when the
selection is finalised, so a clash is refused with its reason instead of being saved.

### Where machine learning is used

| Part | Model / method | Tuned on |
|---|---|---|
| Matching a topic to courses | Every course is split into short pieces (handout topics, objectives, Bulletin text; 15,506 pieces for 2,131 courses). `intfloat/e5-small-v2` embeds them; the 30 closest courses per question are rescored by the cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`. A course's relevance = the mean of its best 3 piece scores, or 1.0 if its title contains the topic. Each topic of a question is scored on its own. | `embedding_eda`: 3 embedders × 3 rerankers (incl. none) on 14 labelled queries; picks the model pair; the cutoff (0.75) is explained under Evaluation |
| Vocabulary gaps ("video editing": no handout uses the words) | **Course neighbours**: the catalogue's best matches for the topic (offered or not, reranker score ≥ 0.3, e.g. GS F343 Short Film and Video Production) are anchors; the offered courses whose content is closest to them (cosine of centred mean piece vectors) fill the places the direct matches leave, at most 3 per topic, labelled as "no direct match; content close to …". No anchor above 0.3 → nothing (cooking, fashion design, marine biology) | anchor floor and cap from a probe of real and nonsense topics |
| One class under two codes (EEE F434 = ECE F434) | Pieces whose embeddings are ≥ 0.99 similar are "twins"; two courses sharing twins for ≥ 90 % of the smaller one's pieces are one class. The student sees one card and takes only one | `equivalence_eda`: precision 0.87, recall 0.62 on 126 listed equivalents |
| Attendance / make-up wording in handouts | local `nli-deberta-v3-small` + MiniLM (extractor only) | hand-checked handouts |

What the LLM does: turns "suggest 2 DELs on VLSI and keep Friday free" into
`get_eligible_courses(category="DEL", about=["VLSI design"], count=2, avoid_day="F")`, and writes the reply.
Guards around it: arguments are validated against the tool schemas (errors go back to the model); a reply naming a
course no tool returned is rewritten once, then those sentences are dropped; if the student named a category the
model didn't search, it is reminded once; no student data is sent to the LLM, only tool results.

## Evaluation

Two separate checks, so a miss can be blamed on the right part.

**Ranking, no LLM** (`python manage.py embedding_eda`, report in `docs/eda/embedding_eda.md`): 14 topic queries
with hand-picked correct courses (`recommender/eda/queries.json`, 3 of them should match nothing), ranked over the
539 offered courses with exactly the app's code, for 3 embedders × (no reranker, ms-marco, bge-reranker-base).

| | hit@5 | MRR | latency / query |
|---|---|---|---|
| **e5-small-v2 + ms-marco reranker (used)** | **0.80** | **0.82** | 0.15 s |
| best without a reranker (bge-small) | 0.79 | 0.84 | 0.01 s |
| e5-small-v2 + bge-reranker-base | 0.63 | 0.78 | 0.77 s |

hit@5 = share of a query's correct courses in its top 5; MRR = 1 / rank of the first correct one. All 3 nonsense
queries find no real match. The F1-best cutoff on these queries is 0.58; the app keeps 0.75 (more precise, and course
neighbours fill a short list), since 14 queries are too few to tune it finely. These numbers are for direct
matches only: "video editing" has no direct match, and its neighbours (Cinematic Art, Mass Media Content & Design,
Reporting and Writing for Media) are checked in the chat evaluation.

**The whole assistant, real LLM** (`python manage.py chat_eval`, report in `docs/eda/chat_eval.md`): 30 messages
from three students (CS, EEE, Mech), including the four examples from the brief, filters, counts, two-category asks,
"I've already finished my HUELs", vocabulary gaps (video editing, journalism) and topics no course covers (cooking,
marine biology). Each message has the tool call a person would make; the pipeline's answer to that call is the
reference, so this measures only the LLM's part. Run of 2026-09-28:

| | |
|---|---|
| tool called with the right category / filters / count / preferences | 29 of 31 calls (the two others were reasonable: "a HUEL" read as count 1; "what suits me" split into one search per category) |
| reference courses that reached the student's cards | 69 of 85 (81 %); the rest the model left out of its answer (e.g. listing 1 of 5 finance OPELs) |
| topics nothing matches: no cards, "nothing matches" said | 3 of 3 |
| courses invented by the model / tool errors | 0 / 0 |
| median time per message | 6.2 s (free tiers; up to ~70 s when every provider is rate-limited and it backs off) |

The first run found two things that were then fixed: the model dropped filters the student asked for ("no
attendance requirement", "open book") in 3 of 30 cases, fixed by naming those phrases in the prompt (0 since); and a
provider answering "200 OK" with an error body crashed the request instead of falling through to the next one.

## Limitations

- **The LLM is the weakest link, and it is a free tier.** It can still pick the wrong category or filter, skip a
  search, or leave a returned course out of its reply; the guards above catch invented courses and missed
  categories, not every misreading. Groq's free tier allows about one question a minute; beyond that the next
  provider answers, so replies can vary in style.
- Ranking quality depends on handout text: 113 of the 539 offered courses have no handout and are matched on their title and
  Bulletin description only. The tuning set is small (14 queries), so the cutoff is a reasonable value, not a
  precise one.
- Course neighbours are "similar content", not "about the topic"; they are labelled that way and ranked after
  direct matches.
- One message at a time: no chat history ("swap the second one" is answered with a question).
- Profile strengths / weaknesses / goals / CGPA are stored but not used for ranking.
- A prerequisite satisfied by a current course blocks the course this semester (the Regulations may allow
  concurrent registration).
- Caches (handout facts, piece index) are per process: after a separate `ingest`, restart the server.
- The first chat message in a process loads the models (~5–8 s).

## Commands

All `.venv/bin/python manage.py ...`:

| Command | What |
|---|---|
| `ingest [--skip-embeddings] [--wipe-students]` | clear and reload the catalog from `dataset/`, then build pieces, embeddings and same-class pairs |
| `build_embeddings` | rebuild pieces, embeddings and content-detected same-class pairs only |
| `embedding_eda [--models …] [--rerankers …] [--candidates …]` | ranking evaluation on `recommender/eda/queries.json` → `docs/eda/embedding_eda.md` |
| `equivalence_eda` | same-class detection thresholds → `docs/eda/equivalence_eda.md` |
| `chat_eval [--only N …] [--pause S]` | the whole assistant with the real LLM on `recommender/eda/chat_cases.json` → `docs/eda/chat_eval.md` |
| `handout_facts_report` | how many courses have each handout fact unknown |
| `llm_smoke_test` | "Say hello" to each provider on its own (checks keys and model names) |

## Data curation notes

Setup is the same venv. The handout extractor runs two small local models (all-MiniLM-L6-v2 and
nli-deberta-v3-small, Apple MPS or CPU, downloaded once; no API calls).

Run order: `.venv/bin/python extractors/timetable.py`, `.venv/bin/python extractors/handouts.py`,
`.venv/bin/python extractors/bulletin.py`. `handouts.py <file.pdf> ...` runs on a few files only and writes
`handouts_sample.json`. The rule-vs-model decision for every handout field, and all thresholds, are in the config
block at the top of `extractors/handouts.py`. Outputs go to `dataset/code processed/`; hand-curated files are in
`dataset/manually processed/`, named after the PDF they came from.

`extractors/handouts.py` is deliberately small (~500 lines): it reads the common handout layouts (labelled header
lines, numbered section headings, ruled plan and evaluation tables) and flags everything else instead of
special-casing it. Attendance and make-up wording is decided by two local models (`handout_models.py`); everything
else is rules. A full run takes about 1.5 minutes.

Each handout record has two lists:

- `issues`: things a person should check. `needs_verification` is true exactly when `issues` is non-empty (3 of 540
  handouts: PDFs that are really another course's handout).
- `notes`: things noticed and handled, so not flagged: `printed_code_differs` (a cross-listed or old/new code whose
  title matches the timetable's title for the file's code), `weightage_in_marks_converted`,
  `no_course_plan_in_handout` (project/thesis/study courses).

Scanned handouts (`348_MAC_F214.pdf`, `362_MATH_F214.pdf`) have no text layer; their pages were typed into
`dataset/manually processed/handouts.json` and go through the same code. `course_no` always comes from the file
name. Handouts the rules couldn't read (113: unusual table layouts, plans without table lines, project-course
templates) were checked by hand. Their values are written in `extractors/build_handout_overrides.py` and
`extractors/handout_overrides_text.py`, built into `dataset/manually processed/handout_overrides.json`
(`.venv/bin/python extractors/build_handout_overrides.py`), and applied by the extractor; those records carry the
note `manually_checked`. Where a handout's own printed weights don't add up to 100, they are kept as printed with
the note `handout_weights_add_up_to: <total>`.

Still flagged: `017_BIO_G523.pdf` (really the BIO F212 handout), `160_CS_F111.pdf` (the CS U111 handout),
`344_INSTR_F491.pdf` (the ECE F366 lab-project handout). The right PDFs are needed for these courses.

### CDCs (bulletin)

Each discipline's CDCs are the `core` courses of its entry in `course_lists` (`dataset/code processed/bulletin.json`),
read by code from the Bulletin's "List of Courses". Every programme has `cdc_lists` naming the list(s) that apply
(dual degrees: one per component). All 28 lists were checked against the "Discipline Core - N Units (M Courses)"
footer of the programme charts; where the Bulletin contradicts itself, `CDC_CORRECTIONS` in `extractors/bulletin.py`
fixes it and the list's `note` says why (Environmental & Sustainability: 13 -> 16 courses from the chart; ECE:
ECE F331 -> ECE F314; Pharmacy: PHA F243 replaced by PHA F215 per the Bulletin's own footnote). The BBA list's
heading is an image in the PDF, so it is named `BUSINESS ADMINISTRATION` from its courses.
