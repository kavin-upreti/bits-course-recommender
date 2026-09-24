manually processed was not extracted using an LLM call (owing to the large wsize of the bulletin and academic regulations) but the data from all the handouts, timetable and a lot of the bulletin was extracted using the code which has been written in extractors


## Data curation notes

Run order: `python extractors/timetable.py`, then `python extractors/handouts.py` (it reads `timetable.json`), then `python extractors/bulletin.py`. Outputs go to `dataset/code processed/`; hand-curated files are in `dataset/manually processed/`, named after the PDF they came from.

Handout edge cases handled in `extractors/handouts.py`:

- **Scanned handouts** (`348_MAC_F214.pdf`, `362_MATH_F214.pdf`, identical files) have no text layer. They were typed in by hand into `dataset/manually processed/handouts.json`; the extractor uses those records instead of parsing the PDFs.
- **Course number inside the handout differs from the file name** (e.g. `017_BIO_G523.pdf` is the BIO F212 handout, `162_CS_F215.pdf` prints "CS F342"): the stored `course_no` is always a number that exists in `timetable.json`. Candidates are the file-name number and every printed number found in the timetable; the one whose timetable title best matches the handout's title wins (ties keep the file name). The printed text is kept in `printed_course_no` and the decision is explained in `note`.
- **Header label variants**: "Course No / Number / Code" and "Course Title / Course Name / Name of the course / Course Number & Title" are all accepted.
- **Headings numbered without a dot** (e.g. `091_CE_G527.pdf`: "1 Course Description:") are recognised, but only for known section headings.
- **Bracketed evaluation cells** (e.g. `039_BITS_F219.pdf`: "[Mid Semester] [30%]") are read by the text fallback.
- **Course plans written as prose** (e.g. `030_BITS_E584.pdf`: "The plan of work ... will be decided by the supervisors") have `topics: null` and are not flagged.

Anything the code still can't read is left `null` with `needs_verification: true` and the reason in `note`; nothing is guessed.
