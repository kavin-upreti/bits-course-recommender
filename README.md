manually processed was not extracted using an LLM call (owing to the large wsize of the bulletin and academic regulations) but the data from all the handouts, timetable and a lot of the bulletin was extracted using the code which has been written in extractors


## Data curation notes

Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` (the handout extractor runs two small local models, all-MiniLM-L6-v2 and nli-deberta-v3-small, on Apple MPS or CPU; they are downloaded once and cached; no API calls).

Run order: `.venv/bin/python extractors/timetable.py`, `.venv/bin/python extractors/handouts.py`, `.venv/bin/python extractors/bulletin.py`. `handouts.py <file.pdf> ...` runs on a few files only and writes `handouts_sample.json`. The rule-vs-model decision for every handout field, and all thresholds, are in the config block at the top of `extractors/handouts.py`. Outputs go to `dataset/code processed/`; hand-curated files are in `dataset/manually processed/`, named after the PDF they came from.

`extractors/handouts.py` is deliberately small (~500 lines): it reads the common handout layouts (labelled header lines, numbered section headings, ruled plan and evaluation tables) and flags everything else instead of special-casing it. Attendance and make-up wording is decided by two local models (`handout_models.py`); everything else is rules. A full run takes about 1.5 minutes.

Each handout record has two lists:

- `issues`: things a person should check. `needs_verification` is true exactly when `issues` is non-empty (3 of 540 handouts: PDFs that are really another course's handout).
- `notes`: things noticed and handled, so not flagged: `printed_code_differs` (a cross-listed or old/new code whose title matches the timetable's title for the file's code), `weightage_in_marks_converted`, `no_course_plan_in_handout` (project/thesis/study courses).

Scanned handouts (`348_MAC_F214.pdf`, `362_MATH_F214.pdf`) have no text layer; their pages were typed into `dataset/manually processed/handouts.json` and go through the same code. `course_no` always comes from the file name. Handouts the rules couldn't read (113: unusual table layouts, plans without table lines, project-course templates) were checked by hand. Their values are written in `extractors/build_handout_overrides.py` and `extractors/handout_overrides_text.py`, built into `dataset/manually processed/handout_overrides.json` (`.venv/bin/python extractors/build_handout_overrides.py`), and applied by the extractor; those records carry the note `manually_checked`. Where a handout's own printed weights don't add up to 100, they are kept as printed with the note `handout_weights_add_up_to: <total>`.

Still flagged: `017_BIO_G523.pdf` (really the BIO F212 handout), `160_CS_F111.pdf` (the CS U111 handout), `344_INSTR_F491.pdf` (the ECE F366 lab-project handout). The right PDFs are needed for these courses.
