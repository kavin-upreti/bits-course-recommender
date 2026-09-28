"""Every tunable number of the recommender. Nothing tunable is hardcoded anywhere else.

LLM_PROVIDERS and the API keys come from .env (recommender/llm.py), not from here. The maximum units per semester
comes from the Rule table (regulations), not from here.
"""

# Embeddings (retrieval, stage C1)
EMBEDDING_MODEL = "intfloat/e5-small-v2"   # chosen by the EDA (section 13); run build_embeddings after changing it
EMBEDDING_BATCH_SIZE = 64
# query / passage prefixes each model was trained with (checked on the model cards, 2026-09-27)
EMBEDDING_PREFIXES = {
    "all-MiniLM-L6-v2": {"query": "", "passage": ""},
    "BAAI/bge-small-en-v1.5": {"query": "Represent this sentence for searching relevant passages: ", "passage": ""},
    "intfloat/e5-small-v2": {"query": "query: ", "passage": "passage: "},
}

# Reranking (stage C2) and the relevance cutoff (C3)
# EDA 2026-09-28 (eda/embedding_eda.md): e5 + this reranker won (hit@5 0.80, MRR 0.82, 0.15 s per query).
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 30                  # courses (by embedding score) that reach the reranker
RERANK_PIECES_PER_COURSE = 3            # best pieces per candidate by embedding, plus its title piece
RERANK_BATCH_SIZE = 64
TITLE_STRONG = 0.8                      # a title scoring this high counts on its own; below, the title is ignored
RELEVANCE_TOP_PIECES = 3                # relevance = mean of a course's best 3 piece scores (one passing mention can't carry it)
RELEVANCE_CUTOFF = 0.75                 # EDA 2026-09-28: F1-best 0.58 on 14 queries; kept higher for precision (neighbours fill short lists)
MAX_QUERY_TOPICS = 8                    # "AI, ML, DL, NLP": each topic scored on its own, best one counts
# Neighbours (ranking.neighbours): a topic with fewer real matches than places is topped up with the courses whose
# content is closest to the catalogue's best matches for it ("anchors"). probe 2026-09-28: every nonsense topic
# (cooking, fashion design, marine biology, cricket) had no catalogue course above 0.1; video editing 0.46,
# cybersecurity 0.56.
NEIGHBOUR_ANCHORS = 3
NEIGHBOURS_PER_TOPIC = 3                # probe 2026-09-28: the 4th-5th for 'video editing' were noise (Advanced Manufacturing)
NEIGHBOUR_ANCHOR_FLOOR = 0.3            # an anchor needs at least this relevance; no anchor -> no neighbours
# probe 2026-09-28: every sensible neighbour was >= 0.40 (video editing -> Cinematic Art 0.47, power systems -> Advanced
# Power Electronics 0.45); inside a branch filter the closest can be far ("maths" + "media" -> Graphs and Networks 0.10).
# Random course pairs: p95 0.28.
NEIGHBOUR_MIN_SIMILARITY = 0.3

# Same class under two codes (recommender/equivalents.py). EDA 2026-09-27 (eda/equivalence_eda.md): best F1 on
# listed pairs (precision 0.87, recall 0.62; the 12 "wrong" pairs above it are renamed titles of the same course)
EQUIVALENT_TWIN_SIM = 0.99              # two pieces this similar are the same sentence
EQUIVALENT_OVERLAP = 0.9                # share of the smaller course's pieces with a twin in the other
EQUIVALENT_MEASURE = "smaller"          # beat "both" at every threshold (F1 0.72 vs 0.47)

# Branch words (eligible.split_branches) -> Course.department codes. Used for get_eligible_courses' `branch` argument
# ("maths courses on probability") and for profile strengths like "maths". Codes checked against the catalogue 2026-09-28.
BRANCH_ALIASES = {
    "math": ["MATH"], "maths": ["MATH"], "mathematics": ["MATH"], "mathematical": ["MATH"],
    "eco": ["ECON"], "econ": ["ECON"], "economics": ["ECON"], "finance": ["FIN", "ECON"],
    "eee": ["EEE"], "electrical": ["EEE"], "electrical and electronics": ["EEE"],
    "electronics": ["EEE", "ECE", "INSTR"], "ece": ["ECE"], "electronics and communication": ["ECE"],
    "instrumentation": ["INSTR"], "electronics and instrumentation": ["INSTR"],
    "cs": ["CS"], "cse": ["CS"], "computer science": ["CS"],
    "mech": ["ME"], "mechanical": ["ME"], "civil": ["CE"], "chemical": ["CHE"], "manufacturing": ["MF"],
    "chemistry": ["CHEM"], "physics": ["PHY"], "bio": ["BIO"], "biology": ["BIO"], "biological sciences": ["BIO"],
    "biotech": ["BIOT"], "biotechnology": ["BIOT"], "pharma": ["PHA"], "pharmacy": ["PHA"],
    "humanities": ["HSS"], "general studies": ["GS"], "management": ["MGTS"],
}
BRANCH_FILLER = {"course", "courses", "elective", "electives", "subject", "subjects", "department", "branch", "engineering"}
# A topic made only of these words ("courses", "electives") says nothing to search for, so it's dropped.
GENERIC_TOPIC_WORDS = {"course", "courses", "elective", "electives", "subject", "subjects"}
MAX_TOPIC_CHARS = 100                   # a longer profile phrase is cut (the tool schema caps topics at the same length)

# Results
MAX_RESULTS = 5                         # courses returned per get_eligible_courses call, unless the student asks for a number
MAX_COUNT = 10                          # the most the student can ask for in one call ("suggest 8 HUELs")
MAX_UNVERIFIED = 5                      # max couldnt_verify courses returned (the rest only counted: saves tokens)
TIE_EPSILON = 0.01                      # scores closer than this count as a tie
CATEGORY_TIE_ORDER = ["DEL", "OPEL", "HUEL"]   # tie-break order
MATCHED_ON_MAX_CHARS = 80
MAX_NOT_OFFERED = 3                     # better-matching courses that aren't offered this semester, named in the result
MAX_NEXT_SEMESTER_WARNINGS = 3          # "you'll be eligible next semester" warnings per call

# Profile boost (eligible.apply_profile). Never removes a course; with a topic it only reorders courses past the cutoff.
# Interests / strengths phrases are searched like topics; a real match adds PROFILE_TEXT_WEIGHT x relevance.
# probe 2026-09-28 (14 labelled queries x 5 profiles, weights 0.05-0.3): hit@5 and MRR exactly unchanged. The centred
# course-vector version missed "programming and algorithms" -> Object Oriented Programming (0.19).
PROFILE_TEXT_WEIGHT = 0.15              # 0.11-0.15: about a strong did-well match, 0.3 x (0.7 - 0.2)
# Did well / struggled with: cosine of centred course vectors (like neighbours), above a floor.
# Two floors, because the probe (2026-09-28, 14 labelled queries x 5 profiles, did well doubled) found: floor 0.3 never
# lowered hit@5 at weights 0.2-0.4 (MRR within +-0.015), floor 0.2 did (hit@5 -0.018, MRR -0.045). Without a topic the
# order was only handout completeness, so there's nothing to make worse, and 0.3 hid real pairs (Signals -> DSP 0.32,
# Programming -> OOP 0.44: most "should match" pairs score 0.25-0.75; random course pairs p90 0.18, p95 0.28).
PROFILE_FLOOR_TOPIC = 0.3               # with a topic: only strong profile matches reorder the real matches
PROFILE_FLOOR = 0.2                     # without a topic: the profile is what orders the list
PROFILE_WEIGHT = 0.3                    # boost = weight x (cosine - floor); weights 0.2-0.4 gave the same probe numbers
GRADE_ORIENTED_FACTOR = 2               # "grades matter a lot": the "did well" part counts this many times

# Evaluation-style dislikes (profile checkboxes): each one found lowers a course by DISLIKE_PENALTY
DISLIKE_PENALTY = 0.05                  # subtracted from the score once per matching dislike
MANY_QUIZZES_THRESHOLD = 4              # "many quizzes" means quiz_count >= this
HEAVY_COMPRE_PERCENT = 45               # "heavy compre" means compre_percent >= this

# Timetable format
EARLY_PERIODS = {1}                     # period 1 = 8 AM
DAY_CODES = ["M", "T", "W", "Th", "F", "S"]   # matches Section.timings (checked: M T W Th F S)
DAY_NAMES = {"M": "Monday", "T": "Tuesday", "W": "Wednesday", "Th": "Thursday", "F": "Friday", "S": "Saturday"}
MAX_PLAN_COURSES = 10                   # courses one check_plan call may add

# Pieces (section 4.2)
PIECE_MIN_WORDS = 2
PIECE_MAX_WORDS = 60

# Agent
MAX_AGENT_ROUNDS = 6
MAX_CHECK_PLAN_CALLS = 3                # per message; each retry resends the whole conversation
LLM_TIMEOUT_SECONDS = 60
LLM_TEMPERATURE = 0.2
LLM_MAX_RETRIES_ON_RATE_LIMIT = 3
LLM_RETRY_BASE_SECONDS = 2              # backoff 2 s, 4 s, 8 s
MAX_MESSAGE_CHARS = 1000
MAX_CARDS = 10
SUMMARY_SENTENCES = 2                   # description sentences in get_course_details and on cards
