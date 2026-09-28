"""The tool definitions sent with every LLM request. Kept short: they cost tokens on every call.
Also the source of truth for argument validation (recommender/validation.py)."""
from . import config

AVOID_8AM = {"type": "boolean", "description": "true if the student wants as few 8 AM classes as possible, false if they say 8 AM is fine. Leave empty if not mentioned."}
AVOID_DAY = {"type": "string", "enum": config.DAY_CODES, "description": "A day the student would like kept free (a preference, not a rule). Leave empty if not mentioned."}

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "get_remaining_requirements",
        "description": "How many HUEL, DEL and OPEL courses and units the student still needs.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_eligible_courses",
        "description": "Eligible courses for the student, best match first (5 per category unless count is given; with several topics, each topic gets its share). Handles eligibility, handout filters, ranking and timetable preferences.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["HUEL", "DEL", "OPEL"],
                             "description": "Leave empty to search every category the student still needs."},
                "about": {"type": "array", "items": {"type": "string", "maxLength": 100}, "maxItems": config.MAX_QUERY_TOPICS,
                          "description": "The student's own topics for this category, one per item, in full words, abbreviations spelled out (e.g. 'ML and NLP' -> ['machine learning', 'natural language processing']). 'AI in electronics' is ONE topic. Don't add topics the student didn't name; leave empty if none."},
                "branch": {"type": "string", "maxLength": 40,
                           "description": "Only when the student limits the search to one subject's courses: 'maths courses on probability' -> branch 'maths', about ['probability']; 'suggest physics courses' -> branch 'physics'. Leave empty when the subject is just a topic ('courses on ML and finance')."},
                "filters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "no_midsem": {"type": "boolean", "description": "Course has no midsem exam."},
                        "no_attendance": {"type": "boolean", "description": "Attendance is not required."},
                        "lenient_makeup": {"type": "boolean", "description": "Makeups allowed for midsem and compre."},
                        "project_based": {"type": "boolean", "description": "Has a project component."},
                        "open_book": {"type": "boolean", "description": "Has at least one open-book exam."},
                        "max_quizzes": {"type": "integer", "minimum": 0, "maximum": 20, "description": "At most this many quizzes. Only if the student gave a number."},
                        "max_compre_percent": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Compre worth at most this %. Only if the student gave a number."},
                        "min_project_percent": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Project worth at least this %. Only if the student gave a number."},
                    },
                },
                "count": {"type": "integer", "minimum": 1, "maximum": config.MAX_COUNT,
                          "description": "How many courses the student asked for in this category, as a number ('three' -> 3). Leave empty if they gave no number."},
                "avoid_8am": AVOID_8AM,
                "avoid_day": AVOID_DAY,
                "exclude": {"type": "array", "items": {"type": "string"}, "maxItems": 20, "description": "Course codes to leave out."},
            },
        },
    },
    {
        "name": "check_plan",
        "description": "Checks whether these courses fit together with the student's current courses in one timetable (clashes, exams, lunch hour, units), and picks sections.",
        "parameters": {
            "type": "object",
            "properties": {
                "courses": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": config.MAX_PLAN_COURSES,
                            "description": "Course codes to check together."},
                "avoid_8am": AVOID_8AM,
                "avoid_day": AVOID_DAY,
            },
            "required": ["courses"],
        },
    },
    {
        "name": "get_course_details",
        "description": "Short details of one course: category for this student, prerequisites, handout facts.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Course code."}}, "required": ["code"]},
    },
]
