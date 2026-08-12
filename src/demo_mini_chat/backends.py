from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from .config import DemoConfig
from .engine import MiniLanguageEngine
from .schemas import CandidateState, GenerationState, ModelInfo, TokenState


TRANSFORMER_MODEL_ID = "datificate/gpt2-small-spanish"
CLASSROOM_CHECKPOINT = Path(".artifacts/mini_language_model.pt")


def _entropy(probabilities: np.ndarray) -> float:
    p = np.clip(probabilities.astype(np.float64), 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())


def _visible_piece(text: str) -> str:
    if text == "":
        return "∅"
    return text.replace(" ", "␠").replace("\n", "↵").replace("\t", "⇥")


class TransformerBackend:
    """Small Spanish GPT-2 used as the realistic next-token demonstrator."""

    def __init__(self, model_id: str = TRANSFORMER_MODEL_ID) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        self.info = ModelInfo(
            kind="transformer",
            name="GPT-2 small · español",
            description="Transformer autoregresivo pequeño: predice el siguiente token a partir del contexto previo.",
            vocabulary_size=int(self.tokenizer.vocab_size),
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            context_window=int(getattr(self.model.config, "n_positions", 1024)),
            training_data="Wikipedia en español · ~3 GB de texto preprocesado",
            tokenizer="Byte Pair Encoding (BPE) · subpalabras/tokens",
            device=str(self.device),
        )

    def _token_state(self, token_id: int) -> TokenState:
        raw = str(self.tokenizer.convert_ids_to_tokens(int(token_id)))
        decoded = self.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        return TokenState(id=int(token_id), raw=raw, display=_visible_piece(decoded))

    def _input_ids(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        if encoded.shape[1] == 0:
            encoded = torch.tensor([[self.tokenizer.eos_token_id]], dtype=torch.long)
        context_window = int(getattr(self.model.config, "n_positions", 1024))
        return encoded[:, -context_window:].to(self.device)

    def step(self, text: str, temperature: float, mode: str, top_k: int) -> GenerationState:
        input_ids = self._input_ids(text)
        with torch.inference_mode():
            output = self.model(input_ids=input_ids)
            logits = output.logits[0, -1].float()
            probabilities = torch.softmax(logits / temperature, dim=-1)

        if mode == "greedy":
            selected_id = int(torch.argmax(probabilities).item())
        else:
            selected_id = int(torch.multinomial(probabilities, num_samples=1).item())

        visible_k = min(top_k, probabilities.shape[0])
        top_probs, top_ids = torch.topk(probabilities, k=visible_k)
        top_probs_np = top_probs.detach().cpu().numpy()
        all_probs_np = probabilities.detach().cpu().numpy()

        candidates: list[CandidateState] = []
        selected_candidate: CandidateState | None = None
        for rank, (token_id, probability) in enumerate(zip(top_ids.tolist(), top_probs_np.tolist(), strict=True), start=1):
            token = self._token_state(int(token_id))
            candidate = CandidateState(**token.model_dump(), probability=float(probability), rank=rank)
            candidates.append(candidate)
            if int(token_id) == selected_id:
                selected_candidate = candidate

        if selected_candidate is None:
            selected_token = self._token_state(selected_id)
            selected_probability = float(all_probs_np[selected_id])
            selected_candidate = CandidateState(
                **selected_token.model_dump(),
                probability=selected_probability,
                rank=int((all_probs_np > selected_probability).sum()) + 1,
            )

        decoded_piece = self.tokenizer.decode([selected_id], clean_up_tokenization_spaces=False)
        text_after = text + decoded_piece

        token_ids = input_ids[0].detach().cpu().tolist()
        input_tokens = [self._token_state(int(token_id)) for token_id in token_ids]
        visible_context = input_tokens[-24:]
        other_mass = max(0.0, 1.0 - float(top_probs_np.sum()))

        return GenerationState(
            state_id=str(uuid4()),
            text_before=text,
            text_after=text_after,
            input_tokens=input_tokens,
            visible_context=visible_context,
            candidates=candidates,
            selected=selected_candidate,
            other_probability_mass=other_mass,
            temperature=temperature,
            mode=mode,
            entropy=_entropy(all_probs_np),
            model=self.info,
        )


class ClassroomBackend:
    """Original semester-V N-gram neural model, kept for comparison and teaching."""

    def __init__(self) -> None:
        self.engine = MiniLanguageEngine.load_or_train(CLASSROOM_CHECKPOINT, config=DemoConfig())
        config = self.engine.config
        self.info = ModelInfo(
            kind="classroom",
            name="Modelo de clase · N-grama neuronal",
            description="La red del semestre V: tres tokens de contexto y una distribución para el siguiente token.",
            vocabulary_size=self.engine.data.size,
            parameter_count=sum(parameter.numel() for parameter in self.engine.model.parameters()),
            context_window=config.window_size,
            training_data="792 frases sintéticas · 5,182 tokens observados · 5,974 ejemplos contexto→token",
            tokenizer="Tokenización simple por palabras/signos",
            device="cpu",
        )

    def _token_state(self, token: str) -> TokenState:
        token_id = int(self.engine.data.word_to_id.get(token, self.engine.data.word_to_id[self.engine.data.UNK]))
        return TokenState(id=token_id, raw=token, display=_visible_piece(token))

    def step(self, text: str, temperature: float, mode: str, top_k: int) -> GenerationState:
        probabilities, _ = self.engine.distribution(text, temperature=temperature)
        ranked_ids = np.argsort(probabilities)[::-1]
        visible_ids = [
            int(idx)
            for idx in ranked_ids
            if self.engine.data.id_to_word[int(idx)] != self.engine.data.SOS
        ][:top_k]

        if mode == "greedy":
            selected_id = int(ranked_ids[0])
        else:
            selected_id = int(np.random.choice(len(probabilities), p=probabilities))

        candidates: list[CandidateState] = []
        selected_candidate: CandidateState | None = None
        for rank, token_id in enumerate(visible_ids, start=1):
            token = self.engine.data.id_to_word[token_id]
            state = self._token_state(token)
            candidate = CandidateState(
                **state.model_dump(), probability=float(probabilities[token_id]), rank=rank
            )
            candidates.append(candidate)
            if token_id == selected_id:
                selected_candidate = candidate

        if selected_candidate is None:
            token = self.engine.data.id_to_word[selected_id]
            state = self._token_state(token)
            selected_probability = float(probabilities[selected_id])
            selected_candidate = CandidateState(
                **state.model_dump(),
                probability=selected_probability,
                rank=int((probabilities > selected_probability).sum()) + 1,
            )

        selected_token = self.engine.data.id_to_word[selected_id]
        text_after = self.engine.append_token(text, selected_token)
        normalized = self.engine.data.normalize_tokens(text)
        input_tokens = [self._token_state(token) for token in normalized]
        other_mass = max(0.0, 1.0 - sum(candidate.probability for candidate in candidates))

        return GenerationState(
            state_id=str(uuid4()),
            text_before=text,
            text_after=text_after,
            input_tokens=input_tokens,
            visible_context=input_tokens[-self.engine.config.window_size :],
            candidates=candidates,
            selected=selected_candidate,
            other_probability_mass=other_mass,
            temperature=temperature,
            mode=mode,
            entropy=_entropy(probabilities),
            model=self.info,
        )


@dataclass(slots=True)
class ModelManager:
    backend: TransformerBackend | ClassroomBackend | None = None
    error: str | None = None

    def load(self, kind: str) -> ModelInfo:
        self.error = None
        try:
            self.backend = TransformerBackend() if kind == "transformer" else ClassroomBackend()
            return self.backend.info
        except Exception as exc:
            self.backend = None
            self.error = f"{type(exc).__name__}: {exc}"
            raise

    @property
    def ready(self) -> bool:
        return self.backend is not None
