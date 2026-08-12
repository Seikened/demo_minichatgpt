from collections import Counter
import re

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-záéíóúüñ]+|[.,!?¿¡]", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class NgramData:
    """Vocabulary and N-gram transformation extracted from the notebook idea."""

    UNK = "<unk>"
    SOS = "<s>"
    EOS = "</s>"

    def __init__(self, order: int, vocab_size: int) -> None:
        if order < 2:
            raise ValueError("order must be >= 2")
        self.order = order
        self.window_size = order - 1
        self.vocab_size = vocab_size
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}

    def fit(self, corpus: list[str]) -> None:
        counts = Counter(token for doc in corpus for token in tokenize(doc))
        reserved = [self.UNK, self.SOS, self.EOS]
        words = [word for word, _ in counts.most_common(self.vocab_size - len(reserved))]
        vocabulary = words + reserved
        self.word_to_id = {word: idx for idx, word in enumerate(vocabulary)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}

    @property
    def size(self) -> int:
        return len(self.word_to_id)

    def normalize_tokens(self, text: str) -> list[str]:
        if not self.word_to_id:
            raise RuntimeError("fit() must be called before encoding text")
        return [token if token in self.word_to_id else self.UNK for token in tokenize(text)]

    def context_tokens(self, text: str) -> list[str]:
        tokens = self.normalize_tokens(text)
        padded = ([self.SOS] * self.window_size) + tokens
        return padded[-self.window_size :]

    def context_ids(self, text: str) -> list[int]:
        return [self.word_to_id[token] for token in self.context_tokens(text)]

    def transform(self, corpus: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not self.word_to_id:
            raise RuntimeError("fit() must be called before transform()")

        x_rows: list[list[int]] = []
        y_rows: list[int] = []

        for doc in corpus:
            tokens = self.normalize_tokens(doc)
            sequence = ([self.SOS] * self.window_size) + tokens + [self.EOS]
            ids = [self.word_to_id[token] for token in sequence]
            for start in range(len(ids) - self.order + 1):
                window = ids[start : start + self.order]
                x_rows.append(window[:-1])
                y_rows.append(window[-1])

        return np.asarray(x_rows, dtype=np.int64), np.asarray(y_rows, dtype=np.int64)

    def state_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "vocab_size": self.vocab_size,
            "word_to_id": self.word_to_id,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "NgramData":
        instance = cls(order=int(state["order"]), vocab_size=int(state["vocab_size"]))
        instance.word_to_id = {str(k): int(v) for k, v in dict(state["word_to_id"]).items()}
        instance.id_to_word = {idx: word for word, idx in instance.word_to_id.items()}
        return instance
