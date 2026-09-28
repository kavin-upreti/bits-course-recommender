# Ideation guide

How we approached the problem, what we tried, what we kept and why. The [README](README.md) says how to run it.

## 1. Starting point

The first sketch was an LLM that turns the student's message into a JSON of wants and don't-wants, then a
recommender that scores courses. Reading the brief again changed the order of work: the hard part isn't the chat, it's
knowing what a student is *allowed* to take. So the data came first, the rules second, and the LLM last.

Ground rules we kept throughout:

- **Never guess.** A value the documents don't state is `null` plus `needs_verification`, never `false`.
  "The handout doesn't say" is shown to the student as "couldn't verify (reason)".
- **Extractors know a document's format, never its content**: no hardcoded course codes, programmes or dates.
  A new semester means new PDFs and a rerun, not new code.
- **Every record keeps its source** (document, page, and for hand-curated rules the quoted text).
- **Anything that decides eligibility is plain Python**, so it can be tested and explained.

## 2. Getting the data out of the PDFs

| Source | How | Result |
|---|---|---|
| Timetable | `pdfplumber` tables; columns found by header name, not position | All 728 entries (sections, rooms, times, exams, instructors); spot-checked exact |
| Bulletin | Page text + tables, split by the document's own headings | 103 programme charts, 28 CDC lists, 23 minors, 136 HUELs, 30 audit courses, 2,006 course descriptions with prerequisites |
| Handouts (540) | Rules for structure (headings, evaluation and plan tables); two small **local** models for policy wording: MiniLM finds the attendance / make-up sentences, `nli-deberta-v3-small` decides what they say | ~425 read fully by code |
| Regulations | Read by hand into `dataset/manually processed/` with quotes (unit caps, lunch hour, higher-degree limit, extra electives, minor rules) | Rules as data, not code |

What had to be done by hand, and is labelled as such:

- **113 handouts** with unusual layouts (unruled or split tables, project templates, schemes written as prose). The
  first extractor tried to handle every layout in code and grew to ~1,500 lines. We replaced it with a ~550-line
  version that handles the common layouts, plus hand-checked values (`extractors/build_handout_overrides.py`). Those
  records carry the note `manually_checked`.
- **2 scanned handouts** with no text layer: typed in, then run through the same code.
- **3 CDC lists where the Bulletin contradicts itself** (Environmental 13 vs 16 courses, ECE F331 vs F314, Pharmacy's
  PHA F243 footnote), fixed in `CDC_CORRECTIONS` with the reason stored. The BBA list heading is an image, so it was
  named by hand.
- **3 handout PDFs are another course's handout** (017_BIO_G523, 160_CS_F111, 344_INSTR_F491). They stay flagged,
  because the fix needs the right PDFs, not code.

Judgement calls: "best 2 of 3 quizzes" stores each quiz at its effective weight; handouts whose printed weights don't
add up to 100 are kept as printed with a note; the course code always comes from the file name, and a different
printed code with a matching title is treated as a cross-listing; a confident "neutral" from the NLI model is stored
as unknown.

## 3. The student's academic state

- **The BITS ID gives the degree.** `2024B3A7PS0832P` = 2024 batch, M.Sc. Economics + B.E. Computer Science,
  Pilani. The Bulletin has a separate chart for each dual pair, so a dual ID maps to that chart.
- **The semester being planned comes from the loaded timetable** (2026-27 Sem 1 → the 2024 batch is going into 3-1).
- **The programme chart fills in compulsory courses**: everything before that semester counts as done, and that
  semester's courses as current. "X or Y" slots ask which one.
- **A course's category depends on the programme** (CDC for one degree, OPEL for another), so it's computed per
  student in a fixed order: audit, GIR, CDC, chart, DEL, HUEL, OPEL.
- **Requirement totals come from real degree audits** (44 courses / 129 units of coursework), because the Bulletin
  edition we have predates the current science foundation. The OPEL count is what's left after GIR, core and DELs.
  A dual degree has no OPEL requirement at all.

## 4. Why not just ask an LLM

A plain LLM call ("here's my profile and the course list, what should I take?") fails the brief in several ways:

- It **invents courses and rules**, and can't know prerequisites, this semester's offerings or a student's chart.
- It **can't verify handout facts**, and it answers "no midsem" confidently whether or not the handout says so.
- **Whole PDFs don't fit**, and big chunks cost thousands of tokens per question on free tiers.
- The **same question gets different answers**, so nothing can be tested.

So the LLM got the two jobs the brief gives it (understanding intent, writing the explanation) and nothing else. It
calls four tools (`get_eligible_courses`, `check_plan`, `get_remaining_requirements`, `get_course_details`) with
arguments; Python does the rest. Guards around it:

- arguments are checked against the tool schemas (sloppy ones like `"None"` or `"3"` are fixed, wrong ones sent back);
- if the student named a category the model didn't search, it's reminded once;
- any course code in the reply that no tool returned triggers a rewrite, then those sentences are removed;
- the cards and the "not shown, and why" list come from the database and tool results, never from the reply text.

Nothing about the student is sent to the LLM, and results are trimmed to what the reply needs. That cut a question
from ~111k tokens (the first version) to 5–14k. To stay free, one small `httpx` client speaks the OpenAI chat format to
Groq, OpenRouter and Gemini and falls through to the next one on a rate limit.

## 5. Matching a topic to courses

- **Pieces, not documents.** Each course is cut into short pieces (title, handout topics, outcomes, description
  sentences; 15.5k in all), so "transformers" matches the one line about transformers instead of being diluted by a
  whole handout.
- **Retrieve, then rerank.** We compared 3 embedding models × (no reranker, ms-marco, bge-reranker) on 14 queries
  with hand-picked answers. `e5-small-v2` + `ms-marco-MiniLM` won (hit@5 0.80, MRR 0.82, 0.2 s). A course's
  relevance is the mean of its best 3 pieces, so one passing mention can't carry it. A course *named* after the
  topic counts fully, because the reranker scores one word against a short title badly.
- **A cutoff, not a top-N.** Only courses at relevance ≥ 0.75 count as matches (F1-best was 0.58, kept higher for
  precision). A short honest list beats five loose ones.
- **Each topic on its own.** "AI, ML, DL, NLP" as one blended embedding half-matches everything. Each topic is
  scored alone and gets its own 30 candidates. Sharing 30 between topics lost Object Oriented Programming for
  "programming, algorithms".
- **Vocabulary gaps.** No handout says "video editing", but the catalogue has a course named after it (not offered).
  Its closest offered courses by content become "no direct match; close to …" neighbours, clearly labelled and ranked
  last. A topic with no catalogue match at all (cooking) returns nothing.
- **Same class, two codes** (EEE F434 = ECE F434): courses sharing near-identical text for ≥ 90% of their pieces are
  one class, so the student sees one card (precision 0.87 on 126 listed equivalents).

Tried and dropped: LLM query expansion and LLM-made "related topics" (drifted, and couldn't be explained), a
hardcoded topic map, one embedding per whole handout.

## 6. Personalisation

The profile has to change the ranking without changing what's allowed, and without making topic searches worse.

| Profile input | What we did | Why |
|---|---|---|
| Interests, strengths | Split into phrases ("programming and algorithms" → two), each searched like a topic; a real match adds a small boost | Comparing the whole text with a course's average content missed obvious pairs (strength "programming" vs OOP scored 0.19) |
| Did well in / struggled with (up to 5 each) | Content similarity to those courses: closer to a good one moves up, to a bad one moves down | Replaces a grade for every course: less typing, stronger signal |
| "Grades matter a lot" | Doubles the "did well" part | For grade-focused students, similar courses are where they're likely to score well |
| Branch words ("maths", "electrical") | A topic that is only a branch word filters by department; as a strength it boosts that department | The search can't read abbreviations; whole-topic only, so "financial markets" is never split |
| SOP plan | A reminder to keep a slot free | Not a ranking signal |
| Weaknesses, job vs research goal, CGPA, per-course grades | Dropped | "Rote learning" can't be read from a handout; a research-vs-job lean couldn't be measured (keyword anchors measure the subject, not the lean; web search and LLM labels were rejected); per-course grades were tedious |

Checked before shipping: the 14 labelled queries under 5 different student profiles. With a topic, the boost only
reorders courses already past the cutoff, with a strict similarity floor (0.3): hit@5 never dropped. Without a topic
there's nothing to make worse, so the floor is 0.2, where real pairs (Signals → DSP 0.32) start to count.

## 7. Timetable intelligence

- **A backtracking search over section choices.** Sections of one course and type with identical times are one
  choice (MATH F211's 12 tutorials shrink a lot), and the course with the fewest options goes first.
- **Hard rules**: no two courses in one slot, a lunch hour free every day, no two exams in one session, the unit cap,
  at most one higher-degree course.
- **Soft preferences** (no 8 AM, a free day, fewest gaps, a teacher) are sort keys, never filters, so there's always
  something to show, with a note when a preference can't be met.
- **Every recommended course is checked against the student's current courses** before it's shown, with the picked
  sections. When nothing fits, the pair that clashes is named ("no lunch hour" vs "clash in every combination").

## 8. How we checked it

- **Unit tests** (183) with fake embedder, reranker and LLM; sockets are blocked, so no test can call an API.
- **`embedding_eda` / `equivalence_eda`**: ranking and same-class detection on labelled data, run on the app's own
  code.
- **`chat_eval`**: real messages with the tool call a person would make. The pipeline's answer to that call is the
  reference, so it measures only the LLM's part.
- **Probes before every tuned number** (floors, cutoffs, caps), recorded next to the number in `recommender/config.py`.

## 9. Not built

Chat history ("swap the second one"), grade prediction, 2+2 international programmes (their rules are in the Bulletin,
but the brief only names single and dual degrees), "courses at the same time as X", and minor-aware ranking.
