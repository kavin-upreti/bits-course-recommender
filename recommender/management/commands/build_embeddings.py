"""`python manage.py build_embeddings`: rebuild every CoursePiece and its embedding (also run at the end of ingest)."""
import time
from collections import Counter

import numpy as np
from django.core.management.base import BaseCommand
from django.db import transaction

from catalog.models import Course, CoursePiece
from recommender import equivalents
from recommender.embeddings import current_model_name, get_embedder
from recommender.piece_index import get_piece_index, invalidate_piece_index
from recommender.pieces import build_pieces


class Command(BaseCommand):
    help = "Cut every course's text into pieces, embed them and store them as CoursePiece rows."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--model", help="sentence-transformers model name (default: config.EMBEDDING_MODEL)")

    def handle(self, *args, **options) -> None:
        started = time.monotonic()
        embedder = get_embedder(options["model"])
        pieces = [(course, kind, text, source) for course in Course.objects.prefetch_related("handouts").order_by("code")
                  for kind, text, source in build_pieces(course)]
        vectors = embedder.embed([text for _, _, text, _ in pieces]) if pieces else np.zeros((0, 0), dtype=np.float32)
        with transaction.atomic():
            CoursePiece.objects.all().delete()
            CoursePiece.objects.bulk_create([
                CoursePiece(course=course, kind=kind, text=text, source=source, embedding=vector.tobytes(),
                            model_name=embedder.model_name)
                for (course, kind, text, source), vector in zip(pieces, vectors)
            ], batch_size=1000)
        invalidate_piece_index()
        self.report(pieces, embedder.model_name, time.monotonic() - started)
        if embedder.model_name == current_model_name():  # why: detection reads the pieces the app ranks with
            pairs = equivalents.detect(get_piece_index())
            equivalents.store(pairs)
            self.stdout.write(f"Same-class courses found from their text: {len(pairs)} pairs")

    def report(self, pieces: list[tuple], model_name: str, seconds: float) -> None:
        """Counts per kind, courses with only a title piece, time taken."""
        per_course = Counter(course.code for course, *_ in pieces)
        title_only = sum(1 for code, count in per_course.items() if count == 1)
        self.stdout.write(f"\nEmbeddings ({model_name}): {Course.objects.count()} courses, {len(pieces)} pieces, "
                          f"{title_only} courses with only a title piece, {seconds:.1f} s")
        for kind, count in sorted(Counter(kind for _, kind, *_ in pieces).items()):
            self.stdout.write(f"  {kind:22} {count:6}")
        if model_name != current_model_name():
            self.stdout.write(f"  note: config.EMBEDDING_MODEL is {current_model_name()}, so ranking won't use these")
