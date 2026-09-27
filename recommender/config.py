"""Every tunable number of the recommender (todo.md section 1). Nothing tunable is hardcoded anywhere else.

GEMINI_API_KEY / GEMINI_MODEL come from .env (recommender/llm.py), not from here. The maximum units per semester
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
BEST_PIECE_WEIGHT = 0.0                 # content = w * best piece + (1 - w) * mean of the best 3; 0 (2026-09-27): one
                                        # passing mention ("applications in NLP") no longer carries a course
TITLE_STRONG = 0.8                      # a title scoring this high counts on its own; below, the title is ignored
DOMAIN_WEIGHT = 0.3                     # the worst-fitting department keeps 70% of its score (ranking_probe 2026-09-27)
DOMAIN_NEUTRAL_DEPARTMENTS = {"BITS"}   # institute-wide interdisciplinary courses: no subject to fit
RELEVANCE_TOP_PIECES = 3
RELEVANCE_CUTOFF = 0.54                 # EDA 2026-09-27 (w=0, title gate, domain): max-F1 threshold; no-match queries 3/3
LOOSE_MATCH_COUNT = 3
LOOSE_MATCH_FLOOR = 0.3                 # probe 2026-09-27: below this "loosely related" was noise (web development -> Issues in Economic Development, 0.02)
MAX_QUERY_TOPICS = 8                    # "AI, ML, DL, NLP": each topic scored on its own, best one counts                   # "loosely_related" courses when nothing passes the cutoff

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
LUNCH_PERIODS = {4, 5, 6}               # at least one of these must be free every day
DAY_CODES = ["M", "T", "W", "Th", "F", "S"]   # matches Section.timings (checked: M T W Th F S)
DAY_NAMES = {"M": "Monday", "T": "Tuesday", "W": "Wednesday", "Th": "Thursday", "F": "Friday", "S": "Saturday"}

# check_plan search
PLAN_SEARCH_NODE_LIMIT = 200_000
MAX_PLAN_COURSES = 10

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
