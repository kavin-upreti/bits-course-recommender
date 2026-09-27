"""`python manage.py handout_facts_report`: how many courses have each handout fact unknown, and the quiz-count rule
applied to real component names."""
from django.core.management.base import BaseCommand

from recommender.handout_facts import facts_report


class Command(BaseCommand):
    help = "Print unknowns per handout fact, evaluation kinds and sample quiz counts."

    def handle(self, *args, **options) -> None:
        self.stdout.write(facts_report())
