from dataclasses import asdict, dataclass


@dataclass(slots=True)
class DemoConfig:
    """Small defaults so the model trains quickly during a live demo."""

    order: int = 4
    vocab_size: int = 512
    embedding_dim: int = 48
    hidden_dim: int = 96
    dropout: float = 0.10
    batch_size: int = 128
    learning_rate: float = 0.08
    max_epochs: int = 80
    patience: int = 10
    seed: int = 1111

    @property
    def window_size(self) -> int:
        return self.order - 1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
