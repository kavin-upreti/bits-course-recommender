"""All CoursePiece embeddings of the current model, in memory (todo.md section 4.5).

Loaded once per process; `invalidate_piece_index` (called by build_embeddings) forces a reload.
"""
from dataclasses import dataclass, field

import numpy as np

from catalog.models import CoursePiece

from .embeddings import current_model_name


@dataclass
class PieceIndex:
    model_name: str
    matrix: np.ndarray                      # (n_pieces, d)
    course_codes: list[str]
    kinds: list[str]
    texts: list[str]
    rows_by_course: dict[str, list[int]] = field(default_factory=dict)
    course_vectors: dict[str, np.ndarray] = field(default_factory=dict)  # code -> centred mean of its pieces, normalised

    def __post_init__(self) -> None:
        for row, code in enumerate(self.course_codes):
            self.rows_by_course.setdefault(code, []).append(row)
        means = {code: self.matrix[rows].mean(axis=0) for code, rows in self.rows_by_course.items()}
        # why centred: every course mean shares a big "university course" direction, which squeezed all similarities
        # into 0.92-0.95 (video editing's 4th neighbour was Advanced Manufacturing); without it they spread 0.3-0.7
        average = np.mean(list(means.values()), axis=0) if means else 0.0
        for code, mean in means.items():
            vector = mean - average
            self.course_vectors[code] = vector / (np.linalg.norm(vector) or 1.0)


def load_piece_index(model_name: str) -> PieceIndex:
    """Every piece stored for `model_name`, ordered by course code, then build order."""
    rows = list(CoursePiece.objects.filter(model_name=model_name).order_by("course__code", "pk")
                .values_list("course__code", "kind", "text", "embedding"))
    vectors = [np.frombuffer(bytes(embedding), dtype=np.float32) for *_, embedding in rows]
    return PieceIndex(model_name=model_name,
                      matrix=np.stack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32),
                      course_codes=[code for code, *_ in rows], kinds=[kind for _, kind, *_ in rows],
                      texts=[text for _, _, text, _ in rows])


_index: PieceIndex | None = None


def get_piece_index() -> PieceIndex:
    """The cached index, rebuilt if the configured model changed."""
    global _index
    if _index is None or _index.model_name != current_model_name():
        _index = load_piece_index(current_model_name())
    return _index


def invalidate_piece_index() -> None:
    """Drop the cached index (pieces were rebuilt)."""
    global _index
    _index = None
