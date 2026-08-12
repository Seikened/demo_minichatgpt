from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import DemoConfig
from .corpus import build_demo_corpus
from .data import NgramData, tokenize
from .model import NeuralLM
from .train import TrainingResult, train_model


@dataclass(frozen=True, slots=True)
class Candidate:
    token: str
    probability: float


@dataclass(frozen=True, slots=True)
class GenerationStep:
    context: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    chosen: Candidate
    neighbors: tuple[tuple[str, float], ...]


class MiniLanguageEngine:
    def __init__(
        self,
        model: NeuralLM,
        data: NgramData,
        config: DemoConfig,
        training: TrainingResult | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config
        self.training = training

    @classmethod
    def build(cls, config: DemoConfig | None = None) -> "MiniLanguageEngine":
        config = config or DemoConfig()
        corpus = build_demo_corpus()
        data = NgramData(order=config.order, vocab_size=config.vocab_size)
        data.fit(corpus)
        result = train_model(corpus, data, config)
        return cls(result.model, data, config, training=result)

    @classmethod
    def load_or_train(
        cls,
        checkpoint: Path,
        config: DemoConfig | None = None,
        retrain: bool = False,
    ) -> "MiniLanguageEngine":
        config = config or DemoConfig()
        if checkpoint.exists() and not retrain:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("config") == config.to_dict():
                data = NgramData.from_state_dict(payload["data"])
                model = NeuralLM(
                    vocab_size=data.size,
                    window_size=config.window_size,
                    embedding_dim=config.embedding_dim,
                    hidden_dim=config.hidden_dim,
                    dropout=config.dropout,
                )
                model.load_state_dict(payload["model"])
                model.eval()
                return cls(model, data, config)

        engine = cls.build(config)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": engine.model.state_dict(),
                "data": engine.data.state_dict(),
                "config": config.to_dict(),
            },
            checkpoint,
        )
        return engine

    def distribution(self, text: str, temperature: float = 1.0) -> tuple[np.ndarray, tuple[str, ...]]:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        context = tuple(self.data.context_tokens(text))
        context_ids = torch.tensor([self.data.word_to_id[token] for token in context], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(context_ids.unsqueeze(0)).squeeze(0)
            probs = F.softmax(logits / temperature, dim=0).cpu().numpy()
        return probs, context

    def next_step(
        self,
        text: str,
        temperature: float = 1.0,
        deterministic: bool = False,
        top_k: int = 6,
    ) -> GenerationStep:
        probs, context = self.distribution(text, temperature=temperature)
        ranked_ids = np.argsort(probs)[::-1]
        visible_ids = [idx for idx in ranked_ids if self.data.id_to_word[int(idx)] != self.data.SOS][:top_k]
        candidates = tuple(
            Candidate(self.data.id_to_word[int(idx)], float(probs[int(idx)])) for idx in visible_ids
        )

        if deterministic:
            chosen_id = int(ranked_ids[0])
        else:
            chosen_id = int(np.random.choice(len(probs), p=probs))

        chosen = Candidate(self.data.id_to_word[chosen_id], float(probs[chosen_id]))
        neighbors = tuple(self.closest_words(chosen.token, k=5))
        return GenerationStep(context=context, candidates=candidates, chosen=chosen, neighbors=neighbors)

    def closest_words(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        if word not in self.data.word_to_id or word in {self.data.SOS, self.data.EOS, self.data.UNK}:
            return []

        word_id = self.data.word_to_id[word]
        with torch.no_grad():
            target = self.model.emb.weight[word_id]
            distances = torch.norm(self.model.emb.weight - target, dim=1)
            ranked = torch.argsort(distances).tolist()

        result: list[tuple[str, float]] = []
        for idx in ranked:
            token = self.data.id_to_word[idx]
            if idx == word_id or token in {self.data.SOS, self.data.EOS, self.data.UNK}:
                continue
            result.append((token, float(distances[idx])))
            if len(result) == k:
                break
        return result

    def append_token(self, text: str, token: str) -> str:
        if token == self.data.EOS:
            return text.strip()
        if token in {".", ",", "!", "?"}:
            return f"{text.rstrip()}{token}"
        return f"{text.rstrip()} {token}".strip()

    def normalized_seed(self, text: str) -> str:
        tokens = tokenize(text)
        return " ".join(tokens) if tokens else "la inteligencia artificial"
