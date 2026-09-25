"""Register -> profile -> electives, against the real ingested catalog.

Run: .venv/bin/python manage.py test students
"""
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

    def profile(self, planning: str, **extra):
        return self.client.post("/profile/", {"planning": planning, "goal": "job", "interests": "ml, finance", **extra})

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
        self.assertRedirects(self.profile("3-1"), "/profile/electives/")
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
        # anything the chart names is compulsory, never an elective (the maths courses fall in no named list)
        self.assertFalse(StudentCourse.objects.filter(source="pattern", category__in=("DEL", "HUEL", "OPEL")).exists())

        choices = elective_choices(student.programme, student.admission_year)
        del_code, opel_code = choices["DEL"][0].code, choices["OPEL"][0].code
        rows = {"s21-TOTAL_FORMS": 1, "s21-INITIAL_FORMS": 0, "s22-TOTAL_FORMS": 1, "s22-INITIAL_FORMS": 0}

        # a DEL typed under OPEL is refused, with the right category named
        response = self.client.post("/profile/electives/", {
            **rows, "s21-0-category": "OPEL", "s21-0-course": del_code, "s21-0-comfort": "4",
        })
        self.assertContains(response, "For your programme it&#x27;s a DEL")

        response = self.client.post("/profile/electives/", {
            **rows,
            "s21-0-category": "DEL", "s21-0-course": del_code.lower(), "s21-0-comfort": "4", "s21-0-grade": "A-",
            "s22-0-category": "OPEL", "s22-0-course": opel_code, "s22-0-comfort": "2",
        })
        self.assertRedirects(response, "/")
        self.assertEqual(
            sorted(StudentCourse.objects.filter(source="user").values_list("semester_taken", "course__code", "comfort")),
            [("2-1", del_code, "4"), ("2-2", opel_code, "2")],
        )
        self.assertEqual(StudentCourse.objects.get(course__code=del_code).category, "DEL")
        self.assertContains(self.client.get("/"), del_code)

        # moving the planned semester back to 2-1 drops both electives (neither is in the past any more)
        self.profile("2-1")
        self.assertFalse(StudentCourse.objects.filter(source="user").exists())
        # any semester can be picked; 3-2 makes 2-1, 2-2 and 3-1 the past elective semesters
        self.assertRedirects(self.profile("3-2"), "/profile/electives/")
        self.assertContains(self.client.get("/profile/electives/"), "<h3>3-1 <span")

    def test_minor_needs_third_year(self):
        self.register()
        response = self.profile("2-1", minor=Minor.objects.first().pk)
        self.assertContains(response, "applies from 3-1 onwards")


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

    def test_current_semester_or_is_picked_on_profile(self):
        self.client.post("/profile/", {"planning": "2-2", "alt:ECON F211": "MGTS F211"})
        self.assertIn("MGTS F211", self.codes())
        self.assertNotIn("ECON F211", self.codes())
        # the other side of the OR still counts as done, so it's never recommended
        from recommender.history import done_codes
        self.assertIn("ECON F211", done_codes(Student.objects.get()))

    def test_past_or_is_picked_on_courses_page(self):
        self.client.post("/profile/", {"planning": "3-1"})
        self.assertIn("ECON F211", self.codes())  # unanswered: the chart's first option
        self.assertContains(self.client.get("/profile/electives/"), 'name="alt:ECON F211"')
        rows = {f"s{s}-{k}": v for s in ("21", "22") for k, v in (("TOTAL_FORMS", 0), ("INITIAL_FORMS", 0))}
        self.assertRedirects(self.client.post("/profile/electives/", {**rows, "alt:ECON F211": "MGTS F211"}), "/")
        self.assertIn("MGTS F211", self.codes())
        self.assertNotIn("ECON F211", self.codes())

    def test_elective_counts_and_minor(self):
        self.client.post("/profile/", {"planning": "3-1"})
        student = Student.objects.get()
        del_code = elective_choices(student.programme, student.admission_year)["DEL"][0].code
        rows = {"s21-TOTAL_FORMS": 1, "s21-INITIAL_FORMS": 0, "s22-TOTAL_FORMS": 0, "s22-INITIAL_FORMS": 0}
        self.client.post("/profile/electives/", {**rows, "s21-0-category": "DEL", "s21-0-course": del_code, "s21-0-comfort": "3"})

        from recommender.requirements import elective_needs, minor_progress
        needs = {need.category: need for need in elective_needs(student)}
        self.assertEqual((needs["DEL"].required_courses, needs["DEL"].done_courses, needs["DEL"].courses_left), (4, 1, 3))
        self.assertEqual((needs["HUEL"].required_courses, needs["HUEL"].courses_left), (3, 3))
        self.assertEqual(needs["OPEL"].required_courses, 5)  # derived from the coursework totals, never below 5

        student.minor = Minor.objects.get(name="Finance")
        student.save()
        progress = minor_progress(student)
        self.assertTrue(progress.core_left)
        self.assertGreaterEqual(progress.electives_left, 2)
        self.assertContains(self.client.get("/"), "For finishing the Finance minor")


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
        self.client.post("/profile/", {"planning": "3-1"})
        load = semester_load(Student.objects.get())
        self.assertEqual(load.max_units, 25)
        self.assertEqual(load.free_units, 25 - load.fixed_units)
        self.assertGreater(load.fixed_units, 0)
        self.assertContains(self.client.get("/"), "units free for electives")

        self.client.post("/logout/")
        self.client.post("/register/", {"name": "N", "username": "2026A7PS0001P", "password1": "a-long-pass-123", "password2": "a-long-pass-123"})
        self.client.post("/profile/", {"planning": "1-1"})
        self.assertIsNone(semester_load(Student.objects.get(user__username="2026A7PS0001P")).max_units)
