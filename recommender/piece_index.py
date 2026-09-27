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
    departments: dict[str, int] = field(default_factory=dict)   # "CS" -> row of department_matrix
    department_matrix: np.ndarray | None = None                 # (n_departments, d), normalised


def department_centroids(index: PieceIndex) -> None:
    """What each department teaches, as one vector: the mean of its courses' mean piece vectors ("CS F429" -> "CS").
    why per course first: a course with 90 pieces would otherwise outweigh ten courses with 5."""
    by_department: dict[str, list[np.ndarray]] = {}
    for code, rows in index.rows_by_course.items():
        by_department.setdefault(code.split()[0], []).append(index.matrix[rows].mean(axis=0))
    if not by_department:
        return
    names = sorted(by_department)
    matrix = np.stack([np.mean(by_department[name], axis=0) for name in names])
    index.department_matrix = (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)
    index.departments = {name: row for row, name in enumerate(names)}


def load_piece_index(model_name: str) -> PieceIndex:
    """Every piece stored for `model_name`, ordered by course code, then build order."""
    rows = list(CoursePiece.objects.filter(model_name=model_name).order_by("course__code", "pk")
                .values_list("course__code", "kind", "text", "embedding"))
    vectors = [np.frombuffer(bytes(embedding), dtype=np.float32) for *_, embedding in rows]
    index = PieceIndex(model_name=model_name,
                       matrix=np.stack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32),
                       course_codes=[code for code, *_ in rows], kinds=[kind for _, kind, *_ in rows],
                       texts=[text for _, _, text, _ in rows])
    for row, code in enumerate(index.course_codes):
        index.rows_by_course.setdefault(code, []).append(row)
    department_centroids(index)
    return index


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
