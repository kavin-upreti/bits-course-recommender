"""Every tunable number of the recommender (todo.md section 1). Nothing tunable is hardcoded anywhere else.

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
# EDA 2026-09-27 (docs/eda/embedding_eda.md), with each topic scored on its own: e5 + this reranker won
# (hit@5 0.85, precision 0.79 at the cutoff). None = relevance from the embedding similarities instead.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 30                  # courses (by embedding score) that reach the reranker
RERANK_PIECES_PER_COURSE = 3            # best pieces per candidate by embedding, plus its title piece
RERANK_BATCH_SIZE = 64
TITLE_STRONG = 0.8                      # a title scoring this high counts on its own; below, the title is ignored
RELEVANCE_TOP_PIECES = 3                # relevance = mean of a course's best 3 piece scores (one passing mention can't carry it)
RELEVANCE_CUTOFF = 0.75                 # EDA 2026-09-27; only a few judged queries, so 0.47-0.75 is within noise
MAX_QUERY_TOPICS = 8                    # "AI, ML, DL, NLP": each topic scored on its own, best one counts
# Neighbours (ranking.neighbours): a topic with fewer real matches than places is topped up with the courses whose
# content is closest to the catalogue's best matches for it ("anchors"). probe 2026-09-28: every nonsense topic
# (cooking, fashion design, marine biology, cricket) had no catalogue course above 0.1; video editing 0.46,
# cybersecurity 0.56.
NEIGHBOUR_ANCHORS = 3
NEIGHBOURS_PER_TOPIC = 3                # probe 2026-09-28: the 4th-5th for 'video editing' were noise (Advanced Manufacturing)
NEIGHBOUR_ANCHOR_FLOOR = 0.3            # an anchor needs at least this relevance; no anchor -> no neighbours

# Same class under two codes (recommender/equivalents.py). EDA 2026-09-27 (docs/eda/equivalence_eda.md): best F1 on
# listed pairs (precision 0.87, recall 0.62; the 12 "wrong" pairs above it are renamed titles of the same course)
EQUIVALENT_TWIN_SIM = 0.99              # two pieces this similar are the same sentence
EQUIVALENT_OVERLAP = 0.9                # share of the smaller course's pieces with a twin in the other
EQUIVALENT_MEASURE = "smaller"          # beat "both" at every threshold (F1 0.72 vs 0.47)

# Results
MAX_RESULTS = 5                         # courses returned per get_eligible_courses call, unless the student asks for a number
MAX_COUNT = 10                          # the most the student can ask for in one call ("suggest 8 HUELs")
MAX_UNVERIFIED = 5                      # max couldnt_verify courses returned (the rest only counted: saves tokens)
TIE_EPSILON = 0.01                      # scores closer than this count as a tie
CATEGORY_TIE_ORDER = ["DEL", "OPEL", "HUEL"]   # tie-break order
MATCHED_ON_MAX_CHARS = 80
MAX_NOT_OFFERED = 3                     # better-matching courses that aren't offered this semester, named in the result
MAX_NEXT_SEMESTER_WARNINGS = 3          # "you'll be eligible next semester" warnings per call

# Evaluation-style dislikes (profile checkboxes) — placeholders, confirm with the user
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
