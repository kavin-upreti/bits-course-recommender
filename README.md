manually processed was not extracted using an LLM call (owing to the large wsize of the bulletin and academic regulations) but the data from all the handouts, timetable and a lot of the bulletin was extracted using the code which has been written in extractors


## Data curation notes

Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` (the handout extractor runs two small local models, all-MiniLM-L6-v2 and nli-deberta-v3-small, on Apple MPS or CPU; they are downloaded once and cached; no API calls).

Run order: `.venv/bin/python extractors/timetable.py`, `.venv/bin/python extractors/handouts.py`, `.venv/bin/python extractors/bulletin.py`. `handouts.py <file.pdf> ...` runs on a few files only and writes `handouts_sample.json`. The rule-vs-model decision for every handout field, and all thresholds, are in the config block at the top of `extractors/handouts.py`. Outputs go to `dataset/code processed/`; hand-curated files are in `dataset/manually processed/`, named after the PDF they came from.

Each handout record has two lists:

- `issues`: things a person should check. `needs_verification` is true exactly when `issues` is non-empty.
- `notes`: things the extractor noticed and handled, so they are *not* flagged: `printed_code_differs` (the handout prints a cross-listed or old/new code and its title matches the timetable's title for the file's code), `weightage_in_marks_converted`, `best_N_of_M_<kind>` (e.g. three 15% quizzes, best two count, so the listed weights add up to 115), `components_without_weight` (e.g. "Assignments (non-evaluative)" while the rest add up to 100), `no_course_plan_in_handout` (project/thesis/study courses, or a plan given only as a link), `topics_from_text_layout` (plan printed without table lines), `pass_fail_course`.

Handout edge cases handled in `extractors/handouts.py`:

- **Scanned handouts** (`348_MAC_F214.pdf`, `362_MATH_F214.pdf`, identical files) have no text layer. Their page text and tables were typed in by hand into `dataset/manually processed/handouts.json`; the extractor runs that transcription through the same rules and models as any PDF.
- **Course number inside the handout differs from the file name**: `course_no` always comes from the file name (suffixes like `-1` stripped). If the printed title also differs from the timetable's title (e.g. `017_BIO_G523.pdf` is really the BIO F212 Microbiology handout) it is a `printed_code_mismatch` issue; otherwise a `printed_code_differs` note.
- **Header variants**: "Course No / Number / Code", "Course Title / Course Name / Name of the course / Title of the Course / Course Number & Title", labels swapped (`269_EEE_G554.pdf`), instructor labels without a colon or inside header tables, headings numbered "1.", "1:", "1. 2.", "4.1" or "1 " (`091_CE_G527.pdf`).
- **Tables**: multi-row headers ("Module" / "Number"), plans continued over pages (with or without a repeated header), evaluation tables that follow the plan directly, schemes split over two tables, thesis handouts that print a mid-semester form before the full scheme.
- **Weight cells**: "1 5 %", "20+10", "10% + 5%", "10x2=20 Marks", "60 /30" under "Marks/Weightage (%)", and sub-scores "10 10".
- **Text-only layouts**: evaluation schemes and course plans printed without table lines are read line by line (`058_BITS_F467.pdf`, `312_GS_F366.pdf`).

What is still flagged (16 records) is either a different course's handout (3), a scheme whose printed weights don't add up to 100 (verified against the PDF text), a two-module scheme split across tables, or a plan laid out in unruled columns that can't be read reliably. Nothing is guessed: unreadable values are left null.
