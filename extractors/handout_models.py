"""Local models used by handouts.py: sentence embeddings (locate) and NLI (decide).

Only used while extracting; the JSON they produce needs no model at runtime.
"""
import logging
from dataclasses import dataclass

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util

log = logging.getLogger(__name__)


@dataclass
class Sentence:
    text: str
    page: int | None


@dataclass
class Located:
    sentence: Sentence
    similarity: float


@dataclass
class Decision:
    value: bool | None
    confidence: float | None  # probability of the winning non-neutral label
    evidence: Sentence | None
    neutral_confidence: float = 0.0  # highest "neutral" probability seen (the text mentions it but doesn't say)


class Models:
    """Loads both models once (cached by Hugging Face under ~/.cache/huggingface) on MPS if available."""

    def __init__(self, embedding_model: str, nli_model: str) -> None:
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        log.info("Loading %s and %s on %s", embedding_model, nli_model, self.device)
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        self.nli = CrossEncoder(nli_model, device=self.device)
        labels = {label.lower(): index for index, label in self.nli.model.config.id2label.items()}
        self.entailment, self.contradiction = labels["entailment"], labels["contradiction"]

    def locate(self, sentences: list[Sentence], description: str, top_k: int) -> list[Located]:
        """The top_k sentences most similar to a short description of the field, best first."""
        if not sentences:
            return []
        query = self.embedder.encode(description, convert_to_tensor=True, normalize_embeddings=True)
        embedded = self.embedder.encode([sentence.text for sentence in sentences], convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(query, embedded)[0].tolist()
        ranked = sorted(zip(sentences, scores), key=lambda pair: pair[1], reverse=True)
        return [Located(sentence, score) for sentence, score in ranked[:top_k]]

    def decide(self, premises: list[Sentence], hypotheses: list[tuple[str, bool | None, bool | None]]) -> Decision:
        """Most confident non-neutral NLI result over every (premise, hypothesis) pair.

        Each hypothesis is (text, value if entailed, value if contradicted); None means that label is ignored
        for it. E.g. ("Make-up is never allowed.", False, True), or ("Attendance is optional.", False, None)
        where a contradiction proves nothing (the model contradicts it for "attendance will be shared").
        why: the small model often says "neutral" to a positive hypothesis for a hedged sentence
        ("No make-up except in case of hospitalization") but confidently contradicts the negative one.
        Neutral pairs never win; their best probability is returned so the caller can tell "the text
        doesn't say" (confident neutral -> null) from "the model is unsure" (uncertain).
        Each premise is a single sentence/clause: multi-sentence premises confuse the small NLI model.
        """
        best = Decision(None, 0.0, None)
        pairs = [(premise, hypothesis) for premise in premises for hypothesis in hypotheses]
        if not pairs:
            return best
        probabilities = self.nli.predict([(premise.text, text) for premise, (text, _, _) in pairs], apply_softmax=True)
        neutral = 0.0
        for (premise, (_, if_entailed, if_contradicted)), scores in zip(pairs, probabilities):
            label = int(scores.argmax())
            value = if_entailed if label == self.entailment else if_contradicted if label == self.contradiction else None
            if value is None:  # neutral, or a label this hypothesis ignores
                neutral = max(neutral, float(scores[label])) if label not in (self.entailment, self.contradiction) else neutral
                continue
            if float(scores[label]) > (best.confidence or 0.0):
                best = Decision(value, float(scores[label]), premise)
        best.neutral_confidence = neutral
        return best
