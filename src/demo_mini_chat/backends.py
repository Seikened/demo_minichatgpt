from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import unicodedata
from uuid import uuid4

import numpy as np
import torch

from .config import DemoConfig
from .engine import MiniLanguageEngine
from .schemas import CandidateState, GenerationState, ModelInfo, TokenState


TRANSFORMER_MODEL_ID = "mrm8488/spanish-gpt2"
CLASSROOM_CHECKPOINT = Path(".artifacts/mini_language_model.pt")


def _entropy(probabilities: np.ndarray) -> float:
    p = np.clip(probabilities.astype(np.float64), 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())


def _visible_piece(text: str) -> str:
    """Human-friendly token label while preserving the raw token separately."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n").replace("ĉ", "\t")
    if text == "":
        return "∅"
    if not text.strip():
        if "\n" in text:
            return "↵"
        if "\t" in text:
            return "⇥"
        return "␠"
    return text.strip()


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
            name="Spanish GPT-2 · ~0.1B",
            description="Transformer autoregresivo pequeño entrenado desde cero en español.",
            vocabulary_size=int(self.tokenizer.vocab_size),
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            context_window=int(getattr(self.model.config, "n_positions", 1024)),
            training_data="large_spanish_corpus · ~20 GB · 95% entrenamiento / 5% validación",
            tokenizer="BPE · subpalabras/tokens",
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

    def _build_state(
        self,
        text_before: str,
        context_ids: list[int],
        probabilities: torch.Tensor,
        selected_id: int,
        temperature: float,
        mode: str,
        top_k: int,
    ) -> GenerationState:
        visible_k = min(top_k, probabilities.shape[0])
        top_probs, top_ids = torch.topk(probabilities, k=visible_k)
        top_probs_list = top_probs.detach().cpu().tolist()
        top_ids_list = top_ids.detach().cpu().tolist()

        candidates: list[CandidateState] = []
        selected_candidate: CandidateState | None = None
        for rank, (token_id, probability) in enumerate(zip(top_ids_list, top_probs_list, strict=True), start=1):
            token = self._token_state(int(token_id))
            candidate = CandidateState(**token.model_dump(), probability=float(probability), rank=rank)
            candidates.append(candidate)
            if int(token_id) == selected_id:
                selected_candidate = candidate

        selected_probability = float(probabilities[selected_id].item())
        if selected_candidate is None:
            selected_token = self._token_state(selected_id)
            selected_candidate = CandidateState(
                **selected_token.model_dump(),
                probability=selected_probability,
                rank=int((probabilities > selected_probability).sum().item()) + 1,
            )

        decoded_piece = self.tokenizer.decode([selected_id], clean_up_tokenization_spaces=False)
        text_after = text_before + decoded_piece
        visible_ids = context_ids[-64:]
        input_tokens = [self._token_state(int(token_id)) for token_id in visible_ids]
        visible_context = input_tokens[-24:]
        other_mass = max(0.0, 1.0 - float(sum(top_probs_list)))
        safe_probs = probabilities.clamp_min(1e-12)
        entropy = float((-(safe_probs * torch.log2(safe_probs))).sum().item())

        return GenerationState(
            state_id=str(uuid4()),
            text_before=text_before,
            text_after=text_after,
            input_tokens=input_tokens,
            visible_context=visible_context,
            candidates=candidates,
            selected=selected_candidate,
            other_probability_mass=other_mass,
            temperature=temperature,
            mode=mode,
            entropy=entropy,
            model=self.info,
        )

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

        return self._build_state(
            text,
            input_ids[0].detach().cpu().tolist(),
            probabilities,
            selected_id,
            temperature,
            mode,
            top_k,
        )

    def iter_generate(
        self,
        text: str,
        temperature: float,
        mode: str,
        top_k: int,
        max_tokens: int,
        stop_strings: list[str] | None = None,
    ) -> Iterator[GenerationState]:
        """Yield tokens as they are produced while preserving the model cache."""
        stop_strings = stop_strings or []
        initial_text = text
        current_text = text
        input_ids = self._input_ids(text)
        context_ids = input_ids[0].detach().cpu().tolist()
        model_input = input_ids
        past_key_values = None
        available_positions = max(0, int(self.info.context_window or 1024) - len(context_ids))

        for _ in range(min(max_tokens, available_positions)):
            with torch.inference_mode():
                output = self.model(
                    input_ids=model_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
                logits = output.logits[0, -1].float()
                probabilities = torch.softmax(logits / temperature, dim=-1)

            if mode == "greedy":
                selected_id = int(torch.argmax(probabilities).item())
            else:
                selected_id = int(torch.multinomial(probabilities, num_samples=1).item())

            state = self._build_state(
                current_text,
                context_ids,
                probabilities,
                selected_id,
                temperature,
                mode,
                top_k,
            )
            yield state
            current_text = state.text_after
            context_ids.append(selected_id)

            if selected_id == self.tokenizer.eos_token_id:
                break
            generated_suffix = current_text[len(initial_text):]
            if any(stop and stop in generated_suffix for stop in stop_strings):
                break

            model_input = torch.tensor([[selected_id]], dtype=torch.long, device=self.device)

    def generate(
        self,
        text: str,
        temperature: float,
        mode: str,
        top_k: int,
        max_tokens: int,
        stop_strings: list[str] | None = None,
    ) -> list[GenerationState]:
        return list(self.iter_generate(text, temperature, mode, top_k, max_tokens, stop_strings))


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
        visible_ids = [int(idx) for idx in ranked_ids if self.engine.data.id_to_word[int(idx)] != self.engine.data.SOS][:top_k]

        if mode == "greedy":
            selected_id = int(ranked_ids[0])
        else:
            selected_id = int(np.random.choice(len(probabilities), p=probabilities))

        candidates: list[CandidateState] = []
        selected_candidate: CandidateState | None = None
        for rank, token_id in enumerate(visible_ids, start=1):
            token = self.engine.data.id_to_word[token_id]
            state = self._token_state(token)
            candidate = CandidateState(**state.model_dump(), probability=float(probabilities[token_id]), rank=rank)
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
        input_tokens = [self._token_state(token) for token in normalized[-64:]]
        other_mass = max(0.0, 1.0 - sum(candidate.probability for candidate in candidates))

        return GenerationState(
            state_id=str(uuid4()),
            text_before=text,
            text_after=text_after,
            input_tokens=input_tokens,
            visible_context=input_tokens[-self.engine.config.window_size:],
            candidates=candidates,
            selected=selected_candidate,
            other_probability_mass=other_mass,
            temperature=temperature,
            mode=mode,
            entropy=_entropy(probabilities),
            model=self.info,
        )

    def iter_generate(
        self,
        text: str,
        temperature: float,
        mode: str,
        top_k: int,
        max_tokens: int,
        stop_strings: list[str] | None = None,
    ) -> Iterator[GenerationState]:
        stop_strings = stop_strings or []
        initial_text = text
        current_text = text
        for _ in range(max_tokens):
            state = self.step(current_text, temperature, mode, top_k)
            yield state
            current_text = state.text_after
            if state.selected.raw == self.engine.data.EOS:
                break
            generated_suffix = current_text[len(initial_text):]
            if any(stop and stop in generated_suffix for stop in stop_strings):
                break

    def generate(
        self,
        text: str,
        temperature: float,
        mode: str,
        top_k: int,
        max_tokens: int,
        stop_strings: list[str] | None = None,
    ) -> list[GenerationState]:
        return list(self.iter_generate(text, temperature, mode, top_k, max_tokens, stop_strings))


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
