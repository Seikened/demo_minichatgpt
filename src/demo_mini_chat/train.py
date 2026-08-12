from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import DemoConfig
from .data import NgramData
from .model import NeuralLM


@dataclass(slots=True)
class TrainingResult:
    model: NeuralLM
    best_loss: float
    epochs: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(corpus: list[str], data: NgramData, config: DemoConfig) -> TrainingResult:
    seed_everything(config.seed)
    x, y = data.transform(corpus)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = NeuralLM(
        vocab_size=data.size,
        window_size=config.window_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    completed_epochs = 0

    for epoch in range(config.max_epochs):
        model.train()
        epoch_losses: list[float] = []
        for context_ids, next_ids in loader:
            logits = model(context_ids)
            loss = criterion(logits, next_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        completed_epochs = epoch + 1
        mean_loss = float(np.mean(epoch_losses))
        if mean_loss < best_loss - 1e-4:
            best_loss = mean_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(model=model, best_loss=best_loss, epochs=completed_epochs)
