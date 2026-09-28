"""Register -> profile -> electives, against the real ingested catalog.

Run: .venv/bin/python manage.py test students
"""
import re
from io import StringIO

from django.core.management import call_command
from django.test import SimpleTestCase, TestCase

from catalog.models import Minor
from recommender.categories import elective_choices

from .bits_id import BitsIdError, parse_bits_id, resolve_programme
from .models import Student, StudentCourse


class ParseIdTests(SimpleTestCase):
    def test_single_and_dual(self):
        single = parse_bits_id("2025a7ps0832p")
        self.assertEqual((single.admission_year, single.first_code, single.second_code, single.campus), (2025, "A7", None, "Pilani"))
        dual = parse_bits_id("2025B3A7PS0832P")
        self.assertEqual((dual.first_code, dual.second_code), ("B3", "A7"))

    def test_rejects(self):
        for bad in ("", "2025A7PS832P", "2025A7A3PS0832P", "2025X9PS0832P", "2025A6PS0832P", "2025A7PS0832Z"):
            with self.subTest(bad=bad), self.assertRaises(BitsIdError):
                parse_bits_id(bad)


class ProfileFlowTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        call_command("ingest", stdout=StringIO())

    def register(self, bits_id: str = "2024B3A7PS0832P"):
        return self.client.post("/register/", {
            "name": "Test Student", "username": bits_id, "password1": "a-long-pass-123", "password2": "a-long-pass-123",
        })

    def profile(self, **extra):
        return self.client.post("/profile/", {"interests": "ml, finance", **extra})

    def test_dual_id_resolves_to_dual_chart(self):
        programme = resolve_programme(parse_bits_id("2024B3A7PS0832P"))
        self.assertEqual((programme.first_component.name, programme.second_component.name), ("M.Sc. Economics", "B.E. Computer Science"))
        with self.assertRaises(BitsIdError):  # the Bulletin has no M.Sc. + B.Pharm. chart
            resolve_programme(parse_bits_id("2024B3A5PS0832P"))

    def test_register_rejects_unknown_combination(self):
        response = self.register("2024B3A5PS0832P")
        self.assertContains(response, "no dual-degree chart")

    def test_full_flow(self):
        self.assertRedirects(self.register(), "/profile/")
        self.assertRedirects(self.profile(), "/profile/electives/")  # 2024 batch + "2026-27 Sem 1" timetable -> 3-1
        student = Student.objects.get()
        self.assertEqual((student.admission_year, student.current_year, student.current_semester), (2024, 3, 1))
        self.assertEqual(student.interests, ["ml", "finance"])

        # compulsory courses inferred from the dual chart: year 1 from the M.Sc. chart, 2-1 onwards from the dual
        inferred = dict(StudentCourse.objects.filter(source="pattern").values_list("course__code", "status"))
        self.assertEqual(inferred.get("MATH F211"), "completed")   # CS, 2-1 (dual chart)
        self.assertIn("BITS F103", inferred)                       # year 1 (first degree's chart)
        self.assertTrue(all(status in ("completed", "current") for status in inferred.values()))
        self.assertNotIn("BITS F221", inferred)                    # PS-I skipped for now
        self.assertTrue(StudentCourse.objects.filter(source="pattern", status="current").exists())
        self.assertFalse(StudentCourse.objects.filter(source="pattern", category="").exists())
        # anything the chart names is compulsory, never an elective
        self.assertFalse(StudentCourse.objects.filter(source="pattern", category__in=("DEL", "HUEL", "OPEL")).exists())

        choices = elective_choices(student.programme, student.admission_year)
        del_code, opel_code = choices["DEL"][0].code, choices["OPEL"][0].code
        rows = {"s21-TOTAL_FORMS": 1, "s21-INITIAL_FORMS": 0, "s22-TOTAL_FORMS": 1, "s22-INITIAL_FORMS": 0}

        # a DEL typed under OPEL is refused, with the right category named
        response = self.client.post("/profile/electives/", {
            **rows, "s21-0-category": "OPEL", "s21-0-course": del_code,
        })
        self.assertContains(response, "For your programme it&#x27;s a DEL")

        response = self.client.post("/profile/electives/", {
            **rows,
            "s21-0-category": "DEL", "s21-0-course": del_code.lower(),
            "s22-0-category": "OPEL", "s22-0-course": opel_code,
        })
        self.assertRedirects(response, "/")
        self.assertEqual(
            sorted(StudentCourse.objects.filter(source="user").values_list("semester_taken", "course__code")),
            [("2-1", del_code), ("2-2", opel_code)],
        )
        self.assertEqual(StudentCourse.objects.get(course__code=del_code).category, "DEL")
        self.assertContains(self.client.get("/"), del_code)

        # a new timetable moves the student: an earlier one (2025-26 Sem 1 -> 2-1) drops both electives, neither
        # being in the past any more; a Sem 2 one (-> 3-2) makes 2-1, 2-2 and 3-1 the past elective semesters
        from catalog.models import Offering
        Offering.objects.update(semester_tag="2025-26 Sem 1")
        self.client.get("/")
        self.assertEqual((Student.objects.get().current_year, Student.objects.get().current_semester), (2, 1))
        self.assertFalse(StudentCourse.objects.filter(source="user").exists())
        Offering.objects.update(semester_tag="2026-27 Sem 2")
        self.assertContains(self.client.get("/profile/electives/"), "<h3>3-1 <span")

    def test_semester_comes_from_batch(self):
        self.register()
        response = self.client.get("/profile/")
        self.assertContains(response, 'Going into <span class="code">3-1</span>')
        self.assertNotContains(response, 'name="planning"')

    def test_minor_by_year(self):
        # 1st year: not asked; 2nd year: aimed for (not registered); 3rd year on: pursued (registered)
        minor = Minor.objects.first()
        for bits_id, has_minor, registered in (("2026A7PS0001P", False, False), ("2025A7PS0002P", True, False), ("2024A7PS0003P", True, True)):
            self.client.post("/logout/")
            self.register(bits_id)
            self.profile(minor=minor.pk)
            student = Student.objects.get(user__username=bits_id)
            self.assertEqual((student.minor is not None, student.minor_registered), (has_minor, registered), bits_id)
        self.assertContains(self.client.get("/"), f"{minor.name} minor</h4>")


class ChoicesAndRequirementsTests(TestCase):
    """CS single degree (2023A7...): its chart has "ECON F211 or MGTS F211" in 2-2."""

    @classmethod
    def setUpTestData(cls):
        call_command("ingest", stdout=StringIO())

    def setUp(self):
        self.client.post("/register/", {
            "name": "CS Student", "username": "2023A7PS0832P", "password1": "a-long-pass-123", "password2": "a-long-pass-123",
        })

    def codes(self) -> set[str]:
        return set(StudentCourse.objects.values_list("course__code", flat=True))

    def test_id_parts_split_practice_school_from_number(self):
        from .views import id_parts
        parts = id_parts("2025B3A7PS0832P", parse_bits_id("2025B3A7PS0832P"))
        self.assertEqual([text for text, _, _ in parts], ["2025", "B3", "A7", "PS", "0832", "P"])
        self.assertEqual(parts[3][1], "practice school")

    def empty_rows(self) -> dict:
        return {f"s{s}-{k}": v for s in ("21", "22", "31", "32") for k, v in (("TOTAL_FORMS", 0), ("INITIAL_FORMS", 0))}

    def test_past_or_is_picked_on_courses_page(self):
        self.client.post("/profile/", {})  # 2023 batch -> 4-1
        self.assertIn("ECON F211", self.codes())  # unanswered: the chart's first option
        self.assertContains(self.client.get("/profile/electives/"), 'name="alt:ECON F211"')
        self.assertRedirects(self.client.post("/profile/electives/", {**self.empty_rows(), "alt:ECON F211": "MGTS F211"}), "/")
        self.assertIn("MGTS F211", self.codes())
        self.assertNotIn("ECON F211", self.codes())
        # the other side of the OR still counts as done, so it's never recommended
        from recommender.history import done_codes
        self.assertIn("ECON F211", done_codes(Student.objects.get()))

    def test_elective_counts_and_minor(self):
        self.client.post("/profile/", {})
        student = Student.objects.get()
        del_code = elective_choices(student.programme, student.admission_year)["DEL"][0].code
        rows = {**self.empty_rows(), "s21-TOTAL_FORMS": 1}
        self.client.post("/profile/electives/", {**rows, "s21-0-category": "DEL", "s21-0-course": del_code})

        from recommender.requirements import elective_needs, minor_progress
        needs = {need.category: need for need in elective_needs(student)}
        self.assertEqual((needs["DEL"].required_courses, needs["DEL"].done_courses, needs["DEL"].courses_left), (4, 1, 3))
        self.assertEqual((needs["HUEL"].required_courses, needs["HUEL"].courses_left), (3, 3))
        self.assertEqual((needs["OPEL"].required_courses, needs["OPEL"].required_units), (5, 15))  # A7 degree audit

        student.minor = Minor.objects.get(name="Finance")
        student.save()
        progress = minor_progress(student)
        self.assertTrue(progress.core_left)
        self.assertGreaterEqual(progress.electives_left, 2)
        self.assertContains(self.client.get("/"), "Finance minor</h4>")


class BatchRulesTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        call_command("ingest", stdout=StringIO())

    def test_2026_only_courses_hidden_from_older_batches(self):
        from catalog.models import Course, Offering
        programme = resolve_programme(parse_bits_id("2024A7PS0001P"))
        offered_to_2025 = {course.code for courses in elective_choices(programme, 2025).values() for course in courses}
        only_2026 = set(Course.objects.filter(only_2026_batch=True).values_list("code", flat=True))
        self.assertTrue(only_2026)
        self.assertFalse(offered_to_2025 & only_2026)
        # a code with a 2026-only com code *and* a normal one is not 2026-only
        self.assertFalse(Course.objects.filter(only_2026_batch=True, offerings__com_code__lt=5000).exists())

    def test_unit_cap(self):
        from recommender.requirements import semester_load
        self.client.post("/register/", {"name": "S", "username": "2024A7PS0001P", "password1": "a-long-pass-123", "password2": "a-long-pass-123"})
        self.client.post("/profile/", {})
        load = semester_load(Student.objects.get())
        self.assertEqual(load.max_units, 25)
        self.assertEqual(load.free_units, 25 - load.fixed_units)
        self.assertGreater(load.fixed_units, 0)
        self.assertContains(self.client.get("/"), "credits free for electives")

        self.client.post("/logout/")
        self.client.post("/register/", {"name": "N", "username": "2026A7PS0001P", "password1": "a-long-pass-123", "password2": "a-long-pass-123"})
        self.client.post("/profile/", {})
        self.assertIsNone(semester_load(Student.objects.get(user__username="2026A7PS0001P")).max_units)


class ScheduleAndCourseTests(TestCase):
    """Semester page (timetables + exams), course pages and the live requirement preview, for a CS 3-1 student."""

    @classmethod
    def setUpTestData(cls):
        call_command("ingest", stdout=StringIO())

    def setUp(self):
        self.client.post("/register/", {"name": "S", "username": "2024A7PS0001P", "password1": "a-long-pass-123", "password2": "a-long-pass-123"})
        self.client.post("/profile/", {})

    def test_generated_timetables_never_clash(self):
        from recommender.timetable import generate, offerings_for
        codes = list(StudentCourse.objects.filter(status="current").values_list("course__code", flat=True))
        timetables = generate(list(offerings_for(codes).values()))
        self.assertTrue(timetables)
        for timetable in timetables:
            owners: dict = {}
            for pick in timetable.picks:
                for slot in pick.slots:
                    self.assertEqual(owners.setdefault(slot, pick.code), pick.code, slot)
            for day in ("M", "T", "W", "Th", "F", "S"):
                self.assertTrue(any((day, period) not in timetable.by_slot for period in (4, 5, 6)), day)
        scores = [timetable.score() for timetable in timetables]
        self.assertEqual(scores, sorted(scores))

    def test_semester_page(self):
        response = self.client.get("/semester/")
        self.assertContains(response, "Possible timetables")
        self.assertContains(response, 'class="slot t-lecture"')
        self.assertContains(response, "Exam calendar")
        self.assertContains(response, 'href="/course/CS%20F351/"')

    def test_course_page(self):
        response = self.client.get("/course/CS%20F213/")
        self.assertContains(response, "Object Oriented Programming")
        self.assertContains(response, "Evaluation")
        self.assertContains(response, "Done in 2-1")
        self.assertEqual(self.client.get("/course/NOPE%20F999/").status_code, 404)

    def test_home_shows_future_semesters_and_you_are_here(self):
        response = self.client.get("/")
        self.assertContains(response, 'id="you-are-here"')
        self.assertContains(response, "4-2")
        self.assertContains(response, "Remaining as per academic requirements")
        self.assertNotContains(response, ">Compulsory<")

    def test_remaining_preview_counts_typed_courses(self):
        from recommender.categories import category_map
        student = Student.objects.get()
        huel = next(code for code, cat in category_map(student.programme).items() if cat == "HUEL")
        before = self.client.post("/profile/electives/remaining/", {}).content.decode()
        after = self.client.post("/profile/electives/remaining/", {"s22-0-course": huel.lower()}).content.decode()
        self.assertIn("0 of 3 courses done", before)
        self.assertIn("1 of 3 courses done", after)

    def test_exam_date_year(self):
        from datetime import date
        from recommender.timetable import exam_date
        self.assertEqual(exam_date("16/12", "2026-27 Sem 1"), date(2026, 12, 16))
        self.assertEqual(exam_date("5/3", "2026-27 Sem 2"), date(2027, 3, 5))
        self.assertIsNone(exam_date("", "2026-27 Sem 1"))


class DualDegreeNeedsTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        call_command("ingest", stdout=StringIO())

    def test_dual_has_no_opel_requirement(self):
        # M.Sc. Physics + B.E. Mechanical: right after registration nothing elective is done
        self.client.post("/register/", {"name": "D", "username": "2025B5A4PS0001P", "password1": "a-long-pass-123", "password2": "a-long-pass-123"})
        self.client.post("/profile/", {})
        from recommender.requirements import elective_needs
        needs = elective_needs(Student.objects.get())
        self.assertEqual([need.category for need in needs], ["HUEL", "DEL", "DEL"])  # no open requirement (audit)
        self.assertTrue(all(need.done_courses == 0 for need in needs if need.category == "DEL"))


class SecondDegreeTests(TestCase):
    """An M.Sc.-only ID (…B5PS…) picks its B.E. on the profile: not asked in year 1, optional in 2-1, required from 2-2."""

    @classmethod
    def setUpTestData(cls):
        call_command("ingest", stdout=StringIO())

    def register(self, bits_id: str) -> None:
        self.client.post("/register/", {"name": "M", "username": bits_id, "password1": "a-long-pass-123", "password2": "a-long-pass-123"})

    def test_optional_in_2_1(self):
        self.register("2025B5PS0001P")
        self.assertContains(self.client.get("/profile/"), "B.E. degree you expect to get (optional)")
        self.client.post("/profile/", {})
        self.assertEqual(Student.objects.get().programme.name, "M.Sc. Physics")

    def test_required_from_2_2(self):
        self.register("2024B5PS0001P")  # 3-1
        self.assertContains(self.client.post("/profile/", {}), "pick your B.E. degree")
        self.client.post("/profile/", {"second_degree_code": "A4"})
        self.assertEqual(Student.objects.get().programme.name, "M.Sc. Physics with B.E. Mechanical")
        self.assertIn("ME F211", set(StudentCourse.objects.values_list("course__code", flat=True)))  # 3-1 is B.E. year

    def test_first_year_is_not_asked(self):
        self.register("2026B5PS0001P")
        self.assertNotContains(self.client.get("/profile/"), 'name="second_degree_code"')

    def test_dual_id_is_not_asked(self):
        self.register("2025B5A4PS0002P")
        self.assertNotContains(self.client.get("/profile/"), 'name="second_degree_code"')

    def test_finished_batch_is_told(self):
        self.register("2019A7PS0001P")
        self.assertContains(self.client.get("/profile/"), "has finished its 4-year programme")


class RecommenderPreferenceTests(TestCase):
    """The profile's recommender defaults: 8 AM, a free day and evaluation styles to avoid (todo section 2)."""

    @classmethod
    def setUpTestData(cls):
        call_command("ingest", "--skip-embeddings", stdout=StringIO())

    def setUp(self):
        self.client.post("/register/", {"name": "P", "username": "2024A7PS0001P", "password1": "a-long-pass-123", "password2": "a-long-pass-123"})

    def test_saves_and_reloads(self):
        self.client.post("/profile/", {"default_avoid_8am": "on", "default_avoid_day": "F",
                                       "avoid_eval_styles": ["many_quizzes", "no_makeup"]})
        student = Student.objects.get()
        self.assertEqual((student.default_avoid_8am, student.default_avoid_day, student.avoid_eval_styles),
                         (True, "F", ["many_quizzes", "no_makeup"]))
        page = self.client.get("/profile/").content.decode()
        self.assertIn('name="default_avoid_8am" id="id_default_avoid_8am" checked', page)
        self.assertIn('<option value="F" selected>Friday</option>', page)
        self.assertIn('value="no_makeup" id="id_avoid_eval_styles_4" checked', page)
        self.assertIn("ranked a little lower, not removed", page)

        self.client.post("/profile/", {"default_avoid_day": ""})
        student.refresh_from_db()
        self.assertEqual((student.default_avoid_8am, student.default_avoid_day, student.avoid_eval_styles), (False, None, []))

    def test_rejects_invalid_values(self):
        response = self.client.post("/profile/", {"default_avoid_day": "Su", "avoid_eval_styles": ["easy_grading"]})
        self.assertContains(response, "Select a valid choice", count=2)
        self.assertFalse(Student.objects.exists())
        response = self.client.post("/profile/", {"avoid_eval_styles": ["closed_book", "closed_book"]})
        self.assertContains(response, "only once")

    def test_did_well_and_struggled_pickers(self):
        # the first screen already lists the chart's courses (from the ID), and a pick there is saved with the profile
        first = self.client.get("/profile/").content.decode()
        self.assertIn('name="did_well"', first)
        self.assertLess(first.index("Grades matter a lot"), first.index("Courses you did well in"))
        picked = re.search(r'name="did_well" value="([^"]+)"', first).group(1)
        self.client.post("/profile/", {"did_well": [picked]})
        student = Student.objects.get()
        self.assertEqual(student.did_well, [picked])
        done = sorted(StudentCourse.objects.filter(student=student, status="completed").values_list("course__code", flat=True))
        self.assertIn(picked, done)
        self.assertContains(self.client.get("/profile/"), f'value="{done[0]}"')
        response = self.client.post("/profile/", {"did_well": done[:6]})
        self.assertContains(response, "Pick at most 5")
        response = self.client.post("/profile/", {"did_well": done[:2], "struggled": done[1:3]})
        self.assertContains(response, f"Picked as both did well and struggled: {done[1]}")
        self.client.post("/profile/", {"did_well": done[:2], "struggled": [done[3]], "grade_oriented": "on"})
        student.refresh_from_db()
        self.assertEqual((student.did_well, student.struggled, student.grade_oriented), (done[:2], [done[3]], True))

    def test_model_save_validates(self):
        from django.core.exceptions import ValidationError
        self.client.post("/profile/", {})
        student = Student.objects.get()
        student.avoid_eval_styles = ["nope"]
        with self.assertRaises(ValidationError):
            student.save()


class TimetableFilterTests(TestCase):
    """Filters above the timetables re-sort them; same-time sections can be switched (students/_timetables.html)."""

    @classmethod
    def setUpTestData(cls):
        call_command("ingest", "--skip-embeddings", stdout=StringIO())

    def setUp(self):
        self.client.post("/register/", {"name": "S", "username": "2024A7PS0001P", "password1": "a-long-pass-123", "password2": "a-long-pass-123"})
        self.client.post("/profile/", {})

    def test_no_8am_puts_fewest_8ams_first_and_form_remembers(self):
        plain = self.client.get("/semester/").context["timetables"]
        sorted_ = self.client.get("/semester/?no_8am=1&free_day=S&avoid=M-2").context["timetables"]
        self.assertEqual(sorted_[0]["score"]["eight_ams"], min(tt["score"]["eight_ams"] for tt in sorted_))
        self.assertLessEqual(sorted_[0]["score"]["eight_ams"], plain[0]["score"]["eight_ams"])
        page = self.client.get("/semester/?no_8am=1&avoid=M-2&free_day=S").content.decode()
        self.assertIn('<option value="S" selected>Saturday</option>', page)
        self.assertNotIn("{#", page)
        self.assertIn('name="no_8am" value="1" checked', page)
        self.assertIn('value="M-2" aria-label="M 9:00" checked', page)

    def test_teacher_filter_and_same_time_switcher(self):
        from recommender.timetable import TimetableFilters, generate, offerings_for
        codes = list(StudentCourse.objects.filter(status="current").values_list("course__code", flat=True))
        offerings = offerings_for(codes)
        course, names = next((code, sorted({n for s in o.sections.all() for n in s.instructors}))
                             for code, o in offerings.items() if len({n for s in o.sections.all() for n in s.instructors}) > 1)
        name = names[-1]
        first = generate(list(offerings.values()), order=TimetableFilters(teachers=[(course, name)]).key)[0]
        self.assertTrue(first.teaches(course, name) or not any(tt.teaches(course, name) for tt in generate(list(offerings.values()), limit=5000)))
        page = self.client.get(f"/semester/?teacher={course}|{name}").content.decode()
        self.assertIn(f'name="teacher" value="{course}|{name}" checked', page)  # the chosen-teacher chip
        self.assertIn(f'data-names=', page)
        self.assertIn('class="alt-choice', page)  # some pick has other sections at the same time

    def test_impossible_filter_is_explained(self):
        response = self.client.get("/semester/?free_day=M")
        best = response.context["timetables"][0]
        mondays = sum(1 for row in best["rows"] for cell in row["cells"][:1] if cell.get("pick"))
        if mondays:  # every CS 3-1 timetable has Monday classes
            self.assertContains(response, "No timetable keeps Monday free")


class SemesterMoveTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        call_command("ingest", "--skip-embeddings", stdout=StringIO())

    def test_added_courses_become_completed_when_the_semester_moves_on(self):
        from catalog.models import Course, Offering
        self.client.post("/register/", {"name": "S", "username": "2024A7PS0001P", "password1": "a-long-pass-123", "password2": "a-long-pass-123"})
        self.client.post("/profile/", {})
        student = Student.objects.get()
        course = Course.objects.filter(code="GS F232").first()
        StudentCourse.objects.create(student=student, course=course, status="current", source="user", semester_taken="3-1")
        Offering.objects.update(semester_tag="2027-28 Sem 1")  # a newer timetable: the 2024 batch is now in 4-1
        self.client.get("/")
        row = StudentCourse.objects.get(course=course)
        self.assertEqual((row.status, row.semester_taken), ("completed", "3-1"))
