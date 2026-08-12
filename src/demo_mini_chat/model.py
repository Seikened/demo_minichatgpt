import torch
from torch import nn
from torch.nn import functional as F


class NeuralLM(nn.Module):
    """Small feed-forward N-gram language model from the original notebook."""

    def __init__(
        self,
        vocab_size: int,
        window_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * window_size, hidden_dim)
        self.drop1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        x = x.reshape(-1, self.window_size * self.embedding_dim)
        hidden = F.relu(self.fc1(x))
        hidden = self.drop1(hidden)
        return self.fc2(hidden)
