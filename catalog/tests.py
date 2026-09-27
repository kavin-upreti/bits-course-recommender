from django.test import SimpleTestCase

from catalog.management.commands.ingest import bulletin_equivalents


class BulletinEquivalentTests(SimpleTestCase):
    def test_codes_before_the_title_only(self):
        text = "Object model; design patterns. Equivalent: CS F213/IS F213: Object Oriented Programming"
        self.assertEqual(bulletin_equivalents(text, "MAC F212"), ["CS F213", "IS F213"])
        self.assertEqual(bulletin_equivalents("Equivalent: EEE F214, INSTR F214 &ECE F214 : Electronic De-", "ECOM F214"),
                         ["EEE F214", "INSTR F214", "ECE F214"])

    def test_syllabus_wording_is_not_an_equivalence(self):
        self.assertEqual(bulletin_equivalents("Transformers: equivalent circuit, EEE F211 style tests", "ECE F211"), [])
        self.assertEqual(bulletin_equivalents("", "X F111"), [])
