"""Test doubles and the test runner (todo.md section 12). Not imported by app code.

- `NoNetworkRunner` (settings.TEST_RUNNER): any socket connect during tests fails, and the process-wide embedder is a
  `FakeEmbedder`, so no test can call Gemini or download a model (ingest builds embeddings with the fake).
- `FakeEmbedder`: fixed vectors for known texts, a hash-based unit vector for anything else.
- `FakeReranker`: fixed scores for known passages, else (1 + cosine of the fake embeddings) / 2; counts its calls.
- `FakeLLM`: returns scripted `LLMResponse`s in order and records every request.
"""
import copy
import hashlib
import socket

import numpy as np
from django.test.runner import DiscoverRunner

from . import embeddings
from .llm import LLMResponse

FAKE_DIMENSIONS = 8


class NetworkUsedInTest(RuntimeError):
    pass


def _refuse_connect(*args, **kwargs) -> None:
    raise NetworkUsedInTest("A test tried to use the network.")


class FakeEmbedder:
    """Deterministic embedder: `vectors` maps text -> vector (normalised here); other texts get a hash vector."""

    def __init__(self, vectors: dict[str, list[float]] | None = None, dimensions: int = FAKE_DIMENSIONS) -> None:
        self.model_name = embeddings.current_model_name()
        self.dimensions = dimensions
        self.vectors: dict[str, np.ndarray] = {}
        self.set(vectors or {})

    def set(self, vectors: dict[str, list[float]]) -> None:
        """Fix the vectors of some texts (shorter vectors are zero-padded, then normalised)."""
        for text, vector in vectors.items():
            padded = np.zeros(self.dimensions, dtype=np.float32)
            padded[:len(vector)] = vector
            self.vectors[text] = _unit(padded)

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        """One L2-normalised row per text (the kind doesn't change the fake vectors)."""
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.stack([self.vectors.get(text, self._hash_vector(text)) for text in texts])

    def _hash_vector(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        return _unit(np.random.default_rng(seed).standard_normal(self.dimensions).astype(np.float32))


class FakeReranker:
    """Deterministic reranker. `scores`: passage text -> score (any query); others from the fake embeddings."""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.model_name = "fake-reranker"
        self.scores = dict(scores or {})
        self.calls: list[list[tuple[str, str]]] = []

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        self.calls.append(list(pairs))
        embedder = embeddings.get_embedder()
        values = []
        for query, passage in pairs:
            if passage in self.scores:
                values.append(self.scores[passage])
            else:
                query_vector, passage_vector = embedder.embed([query, passage])
                values.append((1 + float(query_vector @ passage_vector)) / 2)
        return np.asarray(values, dtype=np.float32)


def _unit(vector: np.ndarray) -> np.ndarray:
    return (vector / np.linalg.norm(vector)).astype(np.float32)


class FakeLLM:
    """Stands in for `llm.chat`: pops the next scripted response (or raises it, if it's an exception)."""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, system: str, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
        self.requests.append({"system": system, "messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class NoNetworkRunner(DiscoverRunner):
    """The project's test runner: no network, fake embeddings."""

    def setup_test_environment(self, **kwargs) -> None:
        super().setup_test_environment(**kwargs)
        self._connect = socket.socket.connect
        socket.socket.connect = _refuse_connect
        embeddings.set_embedder(FakeEmbedder())
        embeddings.set_reranker(FakeReranker())

    def teardown_test_environment(self, **kwargs) -> None:
        socket.socket.connect = self._connect
        embeddings.set_embedder(None)
        embeddings.set_reranker(None)
        super().teardown_test_environment(**kwargs)
