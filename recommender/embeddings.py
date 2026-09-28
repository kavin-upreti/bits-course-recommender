"""Sentence embeddings for retrieval and a cross-encoder for reranking.

Embedding rows are float32 and L2-normalised, so a dot product is the cosine similarity. Each model gets the
query / passage prefix it was trained with (config.EMBEDDING_PREFIXES): pieces are passages, the student's topic
is the query. Models load lazily, once per process and model name. Tests swap in fakes with `set_embedder` /
`set_reranker` (recommender/testing.py).
"""
from typing import Literal, Protocol

import numpy as np

from . import config

Kind = Literal["query", "passage"]


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str], kind: Kind = "passage") -> np.ndarray:
        """Shape (n, d), float32, L2-normalised rows."""
        ...


class Reranker(Protocol):
    model_name: str

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Relevance of each (query, passage) pair, 0-1."""
        ...


def prefix(model_name: str, kind: Kind) -> str:
    """The prefix `model_name` expects for queries or passages ("" if none / unknown model)."""
    return config.EMBEDDING_PREFIXES.get(model_name, {}).get(kind, "")


class SentenceTransformerEmbedder:
    """Wraps sentence_transformers.SentenceTransformer (runs locally; Apple MPS or CPU)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def embed(self, texts: list[str], kind: Kind = "passage") -> np.ndarray:
        """Embed in batches of EMBEDDING_BATCH_SIZE, with the model's prefix, normalised."""
        if self._model is None:
            # why a late import: loading torch takes seconds; only code that embeds pays for it
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        start = prefix(self.model_name, kind)
        vectors = self._model.encode([start + text for text in texts], batch_size=config.EMBEDDING_BATCH_SIZE,
                                     normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32).reshape(len(texts), -1)


class CrossEncoderReranker:
    """Wraps sentence_transformers.CrossEncoder. Both rerankers we use output raw logits (their configs say
    Identity / nothing), so the sigmoid is applied here, exactly once."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """One batched predict over every pair; sigmoid of the logits."""
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        import torch
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        logits = self._model.predict(pairs, batch_size=config.RERANK_BATCH_SIZE, activation_fn=torch.nn.Identity(),
                                     convert_to_numpy=True, show_progress_bar=False)
        return (1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64)))).astype(np.float32)


_embedder_override: Embedder | None = None
_reranker_override: Reranker | None = None
_loaded: dict[tuple[str, str], object] = {}


def current_model_name() -> str:
    """The model the stored pieces and the piece index use."""
    return config.EMBEDDING_MODEL


def set_embedder(embedder: Embedder | None) -> None:
    """Replace the process-wide embedder (tests); None goes back to the real one."""
    global _embedder_override
    _embedder_override = embedder


def set_reranker(reranker: Reranker | None) -> None:
    """Replace the process-wide reranker (tests); None goes back to the real one."""
    global _reranker_override
    _reranker_override = reranker


def get_embedder(model_name: str | None = None) -> Embedder:
    """The process-wide embedder for `model_name` (default: config.EMBEDDING_MODEL), or the test override."""
    if _embedder_override is not None:
        return _embedder_override
    name = model_name or current_model_name()
    return _loaded.setdefault(("embedder", name), SentenceTransformerEmbedder(name))


def get_reranker(model_name: str | None = None) -> Reranker | None:
    """The process-wide reranker (default: config.RERANKER_MODEL; None when reranking is off), or the test override."""
    if _reranker_override is not None:
        return _reranker_override
    name = model_name or config.RERANKER_MODEL
    return _loaded.setdefault(("reranker", name), CrossEncoderReranker(name)) if name else None


def embed_query(text: str) -> np.ndarray:
    """Shape (d,), normalised, with the model's query prefix."""
    return get_embedder().embed([text], kind="query")[0]
