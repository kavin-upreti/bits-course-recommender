"""`python manage.py ingest`: clear the catalog and reload it from the extractor outputs, cross-checking as it goes.

Reads <DATA_DIR>/code processed/{timetable,bulletin,handouts}.json and
<DATA_DIR>/manually processed/{bulletin,Academic-Regulations-2023,timetable}.json (DATA_DIR in settings).
Rerunnable: everything happens in one transaction, so a failed run leaves the old data in place.

Cross-checks (reported, never fatal):
- Bulletin codes that don't resolve to a timetable course, a timetable equivalent or a manual code mapping
  (normal for courses not offered this semester; the list is the thing to eyeball).
- Handouts whose course isn't in the timetable; prerequisite codes with no course; programmes with no CDC list.
- Record counts per model at the end. The full unresolved lists go to <DATA_DIR>/ingest_report.json.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from catalog import models as m

EXAMPLES_SHOWN = 12
EXPECTED_MINORS = 23  # Bulletin 2025-26; a different count after re-extraction means the minors parser broke  # unresolved codes printed per source; the rest are in the report file
DEFAULT_POLICY = re.compile(r"as per (the )?(institute|augs|agsr|academic|university)|(augs|agsr)\w*\s*(division\s*)?(guidelines|rules|norms)"
                            r"|institute (rules|norms|guidelines)|see part[- ]i\b", re.I)
CODE = re.compile(r"^[A-Z]{2,5} [A-Z]\d{3}[A-Z]?(-\d+)?$")
# "Equivalent: CS F213/IS F213: Object Oriented Programming" at the end of a Bulletin description; codes up to the
# next colon (the colon starts the title). Case-sensitive: "equivalent circuit" in a syllabus isn't one.
EQUIVALENT = re.compile(r"\bEquivalent\s*:\s*([^:]*)")
CODE_IN_TEXT = re.compile(r"\b([A-Z]{2,5}) ?([A-Z]\d{3})\b")


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise CommandError(f"Missing input: {path} (run the extractors first; see README)")
    except ValueError as error:
        raise CommandError(f"Not valid JSON: {path} ({error})")


def bulletin_equivalents(description: str, own_code: str) -> list[str]:
    """Codes a Bulletin description names as equivalent ("Equivalent: CS F213/IS F213: ..."), own code left out."""
    codes = [f"{dept} {number}" for match in EQUIVALENT.finditer(description or "")
             for dept, number in CODE_IN_TEXT.findall(match.group(1))]
    return [code for code in dict.fromkeys(codes) if code != own_code]


def department(code: str) -> str:
    return code.split(" ")[0]


def discipline_code(lists: dict, programme: dict) -> str:
    """Course prefix of a single degree ("CS"), read from its CDC list's project courses (XXX F266 -> CS F266)."""
    if programme["type"] != "single" or not programme.get("cdc_lists"):
        return ""
    project_courses = lists[programme["cdc_lists"][0]]["project_courses"]
    return department(project_courses[0]) if project_courses else ""


class Command(BaseCommand):
    help = "Clear and reload all catalog data (courses, timetable, handouts, programmes, rules) with cross-checks."

    def add_arguments(self, parser):
        parser.add_argument("--wipe-students", action="store_true",
                            help="also delete all student data (needed if students exist: they reference catalog rows)")
        parser.add_argument("--skip-embeddings", action="store_true",
                            help="don't rebuild course pieces and their embeddings (build_embeddings) afterwards")

    def handle(self, *args, **options):
        from recommender.handout_facts import clear_facts_cache
        from students.models import Student

        base = Path(settings.DATA_DIR)
        code_dir, manual_dir = base / "code processed", base / "manually processed"
        timetable, bulletin, handouts = (load(code_dir / f"{name}.json") for name in ("timetable", "bulletin", "handouts"))
        manual_bulletin = load(manual_dir / "bulletin.json")
        regulations = load(manual_dir / "Academic-Regulations-2023.json")
        manual_timetable = load(manual_dir / "timetable.json")

        if Student.objects.exists() and not options["wipe_students"]:
            raise CommandError("Students exist and reference catalog rows; rerun with --wipe-students to delete them too.")

        self.report: dict[str, list] = defaultdict(list)
        clear_facts_cache()
        with transaction.atomic():
            if options["wipe_students"]:
                Student.objects.all().delete()
            self.clear()
            courses = self.load_courses(timetable, bulletin, handouts, manual_bulletin)
            self.load_timetable(timetable, courses)
            self.load_handouts(handouts, courses, timetable)
            self.load_bulletin(bulletin, courses, manual_bulletin)
            self.load_rules(manual_bulletin, regulations, manual_timetable)
            self.cross_check(bulletin, timetable, manual_bulletin)

        (base / "ingest_report.json").write_text(json.dumps(self.report, indent=1, ensure_ascii=False))
        self.print_counts()
        self.stdout.write(f"Full cross-check lists: {base / 'ingest_report.json'}")
        if not options["skip_embeddings"]:
            call_command("build_embeddings", stdout=self.stdout)

    # ------------------------------------------------------------ clear
    def clear(self):
        for model in (m.Course, m.Programme, m.Minor, m.AuditCourse, m.CategoryRequirement, m.Rule, m.KnownGap):
            model.objects.all().delete()  # cascades to offerings, sections, handouts, slots, links

    # ------------------------------------------------------------ courses
    def load_courses(self, timetable, bulletin, handouts, manual_bulletin) -> dict[str, m.Course]:
        """One Course per code seen anywhere. Title / units / description from the Bulletin, credits from the timetable."""
        records: dict[str, dict] = {}

        def seen(code: str | None, title: str = "", doc: str = ""):
            if code and CODE.match(code):
                record = records.setdefault(code, {"code": code, "department": department(code), "title": "", "sources": {}})
                record["title"] = record["title"] or (title or "")
                if doc:
                    record["sources"].setdefault("seen_in", [])
                    if doc not in record["sources"]["seen_in"]:
                        record["sources"]["seen_in"].append(doc)

        for d in bulletin["course_descriptions"]:
            seen(d["code"], d["title"], "bulletin.pdf")
            records[d["code"]].update(L=d["L"], P=d["P"], units=d["U"], description=d["description"] or "",
                                      prerequisites=d["prerequisites"], needs_verification=d["needs_verification"],
                                      note=d.get("note") or "")
            records[d["code"]]["sources"]["description"] = d["source"]
        for c in timetable["courses"]:
            seen(c["course_no"], c["title"].title() if c["title"].isupper() else c["title"], "timetable.pdf")
            credits, record = c["credits"], records[c["course_no"]]
            # why AND: a code can have several timetable entries (a 2026-only com code next to a normal one);
            # it's 2026-only only if every entry is
            only_2026 = record.get("only_2026_batch", True) and not c["allowed_for_before_2026_batch"]
            record.update(com_code=c["com_code"], only_2026_batch=only_2026,
                          T=credits.get("tutorial"), S=credits.get("self_study"))
            for key, field in (("lecture", "L"), ("practical", "P"), ("units", "units")):
                if record.get(field) is None:
                    record[field] = credits.get(key)
            record["sources"]["timetable"] = {"doc": "timetable.pdf", "pages": c.get("source_pages")}
        for course_list in bulletin["course_lists"]:
            for course in course_list["core"] + course_list["discipline_electives"]:
                seen(course["code"], course["title"], "bulletin.pdf")
            for code in course_list["project_courses"]:
                seen(code, "", "bulletin.pdf")
        for programme in bulletin["programmes"]:
            for chart in programme["semesters"]:
                for slot in chart["slots"]:
                    seen(slot["code"], slot["title"] or "", "bulletin.pdf")
                    for alt in slot["alternatives"]:
                        seen(alt, "", "bulletin.pdf")
        for minor in bulletin["minors"]:
            for course in minor["core"] + minor["electives"]:
                seen(course["code"], course["title"], "bulletin.pdf")
        for course in bulletin["huel_pool"]:
            seen(course["code"], course["title"], "bulletin.pdf")
        for handout in handouts:
            seen(handout["course_no"], handout.get("title") or "", "handouts")
        for mapping in manual_bulletin["code_mappings"]:
            seen(mapping["to"], mapping.get("title") or "", "code_mappings")
        for requirement in manual_bulletin["category_requirements"]:
            for course in requirement.get("named_courses") or []:
                seen(course["code"], course.get("title") or "", "bulletin.pdf")
                for alt in course.get("alternatives") or []:
                    seen(alt, "", "bulletin.pdf")

        # why: project courses are XXX F266 / F366 / ... for every degree, so match on the number part only
        project_numbers = {p["code"].split(" ")[1] for p in bulletin["project_course_patterns"]}
        for record in records.values():
            number = record["code"].split(" ")[1]
            record["is_project_course"] = number in project_numbers
            record["is_higher_degree"] = number.startswith("G")

        fields = {f.name for f in m.Course._meta.fields}
        objects = [m.Course(**{k: v for k, v in r.items() if k in fields and v is not None}) for r in records.values()]
        m.Course.objects.bulk_create(objects, batch_size=500)
        return {course.code: course for course in m.Course.objects.all()}

    # ------------------------------------------------------------ timetable
    def load_timetable(self, timetable, courses):
        equivalents, sections = [], []
        for c in timetable["courses"]:
            course = courses[c["course_no"]]
            equivalents += [m.CourseEquivalent(course=course, equivalent_code=code) for code in c.get("equivalents") or []]
            offering = m.Offering.objects.create(
                course=course, semester_tag=timetable["semester_tag"], campus=timetable["campus"], com_code=c["com_code"],
                midsem=c.get("midsem") or "", midsem_date=c.get("midsem_date") or "", midsem_session=c.get("midsem_session") or "",
                compre=c.get("compre") or "", compre_date=c.get("compre_date") or "", compre_session=c.get("compre_session") or "",
                instructor_in_charge=c.get("instructor_in_charge") or "", sources={"timetable": {"pages": c.get("source_pages")}},
                needs_verification=bool(c.get("needs_verification")), note=c.get("note") or "")
            for section_type, by_id in (c.get("sections") or {}).items():
                for section_id, s in by_id.items():
                    sections.append(m.Section(offering=offering, section_id=section_id, type=section_type,
                                              instructors=s.get("instructors") or [], room=s.get("room") or "",
                                              timings=s.get("timings") or {}, cancelled=bool(s.get("cancelled"))))
        m.CourseEquivalent.objects.bulk_create(equivalents, ignore_conflicts=True)
        m.Section.objects.bulk_create(sections, batch_size=1000)

    # ------------------------------------------------------------ handouts
    def load_handouts(self, handouts, courses, timetable):
        offered = {c["course_no"] for c in timetable["courses"]}
        rows = []
        for h in handouts:
            evaluation = h["evaluation"] or []
            weight = lambda kinds: round(sum(c["weightage_percent"] or 0 for c in evaluation if c["kind"] in kinds), 2)
            has = lambda kind: any(c["kind"] == kind for c in evaluation) if evaluation else None
            attendance, makeup = h["attendance"], h["makeup"]
            if h["course_no"] not in offered:
                self.report["handout course not in timetable"].append(h["file"])
            rows.append(m.Handout(
                course=courses[h["course_no"]], file=h["file"], description=h.get("description") or "", topics=h["topics"],
                objectives=h.get("objectives") or [], learning_outcomes=h.get("outcomes") or [],
                topic_groups=h.get("topic_groups") or [],
                evaluation=evaluation, has_midsem=has("midsem"), has_project=has("project"), has_quiz=has("quiz"),
                open_book_percent=round(sum(c["weightage_percent"] or 0 for c in evaluation if c["nature"] == "OB"), 2) if evaluation else None,
                project_percent=weight({"project"}) if evaluation else None, compre_percent=weight({"compre"}) if evaluation else None,
                attendance_text=attendance.get("text") or "", attendance_required=attendance.get("required"),
                attendance_percent=attendance.get("percent"),
                attendance_follows_default=bool(DEFAULT_POLICY.search(attendance.get("text") or "")),
                makeup_text=makeup.get("text") or "", makeup_allowed=makeup.get("allowed"),
                makeup_per_component=makeup.get("per_component") or {},
                makeup_follows_default=bool(DEFAULT_POLICY.search(makeup.get("text") or "")),
                sources={"doc": h["file"], "pages": h.get("source_pages"), "methods": h.get("extraction_methods")},
                needs_verification=h["needs_verification"], note="; ".join(h.get("issues", []) + h.get("notes", []))))
        m.Handout.objects.bulk_create(rows, batch_size=200)

    # ------------------------------------------------------------ bulletin
    def load_bulletin(self, bulletin, courses, manual_bulletin):
        mapped = {x["from"]: x["to"] for x in manual_bulletin["code_mappings"]}
        lists = {cl["discipline"]: cl for cl in bulletin["course_lists"] if cl["discipline"]}

        programmes = {}
        for p in bulletin["programmes"]:
            footer = p.get("footer") or {}
            programmes[p["name"]] = m.Programme.objects.create(
                name=p["name"], degree=p.get("degree") or "", type=p["type"], edition=p.get("edition") or "",
                discipline_code=discipline_code(lists, p),
                batch_range=p.get("batch_range"), core_units=footer.get("core_units"), core_courses=footer.get("core_courses"),
                del_units=footer.get("del_units"), del_courses=footer.get("del_courses"), summer=p.get("summer"),
                final_year_options=p.get("final_year_options") or [], sources={"chart": p.get("source")},
                needs_verification=p.get("needs_verification", True), note=p.get("note") or "")
        # dual degrees: each component is the single-degree programme that uses the same CDC list
        single_by_list = {}
        for p in bulletin["programmes"]:
            if p["type"] == "single" and p.get("cdc_lists"):
                single_by_list.setdefault(p["cdc_lists"][0], programmes[p["name"]])
        for p in bulletin["programmes"]:
            if p["type"] == "dual" and len(p.get("cdc_lists") or []) == 2:
                programme = programmes[p["name"]]
                programme.first_component, programme.second_component = (single_by_list.get(name) for name in p["cdc_lists"])
                programme.save(update_fields=["first_component", "second_component"])
            if p["type"] != "dual_template" and not p.get("cdc_lists"):
                self.report["programme with no CDC list"].append(p["name"])

        slots, links = [], []
        for p in bulletin["programmes"]:
            programme = programmes[p["name"]]
            for chart in p["semesters"]:
                for slot in chart["slots"]:
                    slots.append(m.PatternSlot(
                        programme=programme, year=chart["year"], semester=chart["semester"], slot_type=slot["slot_type"],
                        course=courses.get(slot["code"]), alternatives=slot["alternatives"],
                        elective_category=slot.get("category") or "", units_text=slot.get("units_text") or "",
                        semester_unit_total=chart.get("unit_total") or ""))
            for component, discipline in enumerate(p.get("cdc_lists") or [], 1):
                course_list = lists[discipline]
                entries = [("CDC", c) for c in course_list["core"]] + [("DEL", c) for c in course_list["discipline_electives"]] \
                    + [("project-DEL", {"code": code, "alternatives": [], "pool": None}) for code in course_list["project_courses"]]
                for category, c in entries:
                    code = mapped.get(c["code"], c["code"])
                    if code not in courses:
                        continue
                    links.append(m.ProgrammeCourse(
                        programme=programme, course=courses[code], category=category, component=component,
                        track_or_pool=c.get("pool") or "", alternative_group=c.get("alternatives") or [],
                        inferred=code != c["code"], sources={"list": course_list.get("source"), "listed_code": c["code"]}))
        m.PatternSlot.objects.bulk_create(slots, batch_size=1000)
        m.ProgrammeCourse.objects.bulk_create(links, batch_size=1000)

        m.HuelPoolCourse.objects.bulk_create([m.HuelPoolCourse(course=courses[c["code"]]) for c in bulletin["huel_pool"]
                                              if c["code"] in courses], ignore_conflicts=True)
        m.AuditCourse.objects.bulk_create([m.AuditCourse(code=c["code"], title=c.get("title") or "") for c in bulletin["audit_courses"]],
                                          ignore_conflicts=True)
        minor_courses = []
        for minor in bulletin["minors"]:
            row = m.Minor.objects.create(name=minor["name"], description=minor.get("description") or "",
                                         min_courses=minor.get("min_courses"), min_units=minor.get("min_units"),
                                         exclusion_text=minor.get("exclusion_text") or "", sources={"minor": minor.get("source")},
                                         needs_verification=minor.get("needs_verification", False), note=minor.get("note") or "")
            minor_courses += [m.MinorCourse(minor=row, course=courses[c["code"]], role=role)
                              for role, group in (("core", minor["core"]), ("elective", minor["electives"]))
                              for c in group if c["code"] in courses]
        m.MinorCourse.objects.bulk_create(minor_courses)

        m.GirCourse.objects.bulk_create([m.GirCourse(
            course=courses[c["code"]], heading=r["category"], alternative_group=c.get("alternatives") or [],
            sources={"requirement": r.get("source")})
            for r in manual_bulletin["category_requirements"] for c in r.get("named_courses") or []])
        m.CourseEquivalent.objects.bulk_create([
            m.CourseEquivalent(course=courses[d["code"]], equivalent_code=code, source="bulletin")
            for d in bulletin["course_descriptions"] if d["code"] in courses
            for code in bulletin_equivalents(d["description"], d["code"])], ignore_conflicts=True)
        m.CodeMapping.objects.bulk_create([m.CodeMapping(
            from_code=x["from"], to_course=courses[x["to"]], inferred=x.get("inferred", True), reason=x.get("reason") or "",
            needs_verification=x.get("needs_verification", True), sources={"mapping": x.get("id")})
            for x in manual_bulletin["code_mappings"]])

    # ------------------------------------------------------------ hand-curated rules
    def load_rules(self, manual_bulletin, regulations, manual_timetable):
        m.CategoryRequirement.objects.bulk_create([m.CategoryRequirement(
            category=r["category"], code=r.get("code") or "", group=r.get("group") or "", min_units=r.get("min_units"),
            max_units=r.get("max_units"), min_courses=r.get("min_courses"), max_courses=r.get("max_courses"),
            named_courses=r.get("named_courses") or [], sources={"requirement": r.get("source")},
            needs_verification=r.get("needs_verification", False), note=r.get("note") or "")
            for r in manual_bulletin["category_requirements"]])

        rules = []
        for key in ("dual_degree_rules", "project_course_rules", "huel_rules", "minor_rules"):
            rules += manual_bulletin[key]
        rules += regulations["rules"] + manual_timetable["rules"]
        # the timetable legend (periods, days, exam sessions, credit columns, com-code rule) is stored as rules too
        rules += [dict(manual_timetable[key], group="timetable") for key in
                  ("periods", "days", "midsem_sessions", "compre_sessions", "credit_columns", "com_code_rule")]
        m.Rule.objects.bulk_create([m.Rule(
            rule_id=r["id"], group=r.get("group") or "", description=r.get("description") or "", values=r.get("values") or {},
            quote=(r.get("source") or {}).get("quote") or "", source_doc=(r.get("source") or {}).get("doc") or "",
            applies_to_batches=r.get("applies_to_batches"), sources={"rule": r.get("source")},
            needs_verification=r.get("needs_verification", False), note=r.get("note") or "") for r in rules])

        gaps = manual_bulletin["known_gaps"] + manual_timetable["known_gaps"]
        m.KnownGap.objects.bulk_create([m.KnownGap(
            gap_id=g["id"], description=g["description"], affected=g.get("affected") or {}, user_message=g.get("user_message"),
            sources={"gap": g.get("source")}, needs_verification=g.get("needs_verification", False), note=g.get("note") or "")
            for g in gaps])

    # ------------------------------------------------------------ cross-checks
    def cross_check(self, bulletin, timetable, manual_bulletin):
        # "BITS F101-1" in the timetable is the Bulletin's BITS F101: compare without the "-1" suffix
        offered = {c["course_no"].split("-")[0] for c in timetable["courses"]}
        equivalents = {code.split("-")[0] for c in timetable["courses"] for code in c.get("equivalents") or []}
        mapped = {x["from"] for x in manual_bulletin["code_mappings"]}
        resolves = lambda code: code in offered or code in equivalents or code in mapped

        sources = {
            "programme charts": {slot["code"] for p in bulletin["programmes"] for s in p["semesters"] for slot in s["slots"] if slot["code"]},
            "CDC lists": {c["code"] for cl in bulletin["course_lists"] for c in cl["core"]},
            "discipline electives": {c["code"] for cl in bulletin["course_lists"] for c in cl["discipline_electives"]},
            "minors": {c["code"] for mi in bulletin["minors"] for c in mi["core"] + mi["electives"]},
            "HUEL pool": {c["code"] for c in bulletin["huel_pool"]},
        }
        self.stdout.write("\nBulletin codes that don't resolve to a timetable course, equivalent or mapping "
                          "(usually: not offered this semester):")
        for name, codes in sources.items():
            missing = sorted(code for code in codes if not resolves(code))
            self.report[f"unresolved: {name}"] = missing
            shown = ", ".join(missing[:EXAMPLES_SHOWN]) + (" ..." if len(missing) > EXAMPLES_SHOWN else "")
            self.stdout.write(f"  {name:22} {len(missing):4} of {len(codes):4}  {shown}")

        known = set(m.Course.objects.values_list("code", flat=True))
        prereq_codes = {code for d in bulletin["course_descriptions"] for group in d["prerequisites"] or [] for code in group}
        self.report["prerequisite code with no course"] = sorted(prereq_codes - known)
        self.report["single degree with no discipline code"] = sorted(
            m.Programme.objects.filter(type="single", discipline_code="").values_list("name", flat=True))
        for label in ("handout course not in timetable", "programme with no CDC list", "prerequisite code with no course",
                      "single degree with no discipline code"):
            items = self.report.get(label, [])
            self.stdout.write(f"  {label:34} {len(items):4}  {', '.join(items[:EXAMPLES_SHOWN])}")

        self.timetable_checks(timetable, bulletin)

    def timetable_checks(self, timetable, bulletin):
        """Facts timetable intelligence relies on (ideation 2.1 / Part 8), plus the expected minor count."""
        courses = timetable["courses"]
        sections = [s for c in courses for group in (c.get("sections") or {}).values() for s in group.values()]
        practicals = [s for c in courses for s in (c.get("sections") or {}).get("practical", {}).values()]
        multi_period = sum(any(len(periods) > 1 for periods in s["timings"].values()) for s in practicals)
        no_timings = sum(not s.get("timings") and not s.get("cancelled") for s in sections)
        no_sections = [c["course_no"] for c in courses if not c.get("sections")]
        self.report["timetable: courses with no sections"] = no_sections
        self.stdout.write("\nTimetable checks:")
        self.stdout.write(f"  course entries / unique codes      {len(courses)} / {len({c['course_no'] for c in courses})}")
        self.stdout.write(f"  sections / cancelled               {len(sections)} / {sum(bool(s.get('cancelled')) for s in sections)}")
        self.stdout.write(f"  practicals spanning >1 period      {multi_period} of {len(practicals)}"
                          + ("   <- expected most labs to be multi-period" if practicals and multi_period == 0 else ""))
        self.stdout.write(f"  live sections with no fixed slot   {no_timings}   (project / thesis-style courses)")
        self.stdout.write(f"  courses with no sections           {len(no_sections)}")
        minors = len(bulletin["minors"])
        self.stdout.write(f"  minors                             {minors}" + ("" if minors == EXPECTED_MINORS else
                                                                          f"   <- expected {EXPECTED_MINORS}"))

    def print_counts(self):
        self.stdout.write("\nRecords per model:")
        for model in (m.Course, m.CourseEquivalent, m.Offering, m.Section, m.Handout, m.Programme, m.PatternSlot,
                      m.ProgrammeCourse, m.GirCourse, m.CodeMapping, m.HuelPoolCourse, m.AuditCourse, m.Minor, m.MinorCourse, m.CategoryRequirement,
                      m.Rule, m.KnownGap):
            total = model.objects.count()
            flagged = model.objects.filter(needs_verification=True).count() if hasattr(model, "needs_verification") else None
            self.stdout.write(f"  {model.__name__:20} {total:6}" + (f"   ({flagged} flagged)" if flagged else ""))
