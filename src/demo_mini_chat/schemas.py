from typing import Literal

from pydantic import BaseModel, Field


DecodeMode = Literal["sample", "greedy"]
ModelKind = Literal["transformer", "classroom"]


class ModelLoadRequest(BaseModel):
    model: ModelKind = "transformer"


class TokenState(BaseModel):
    id: int
    raw: str
    display: str


class CandidateState(TokenState):
    probability: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)


class ModelInfo(BaseModel):
    kind: ModelKind
    name: str
    description: str
    vocabulary_size: int
    parameter_count: int | None = None
    context_window: int | None = None
    training_data: str
    tokenizer: str
    device: str


class StepRequest(BaseModel):
    text: str = Field(min_length=1)
    temperature: float = Field(default=0.85, gt=0.0, le=3.0)
    mode: DecodeMode = "sample"
    top_k: int = Field(default=64, ge=12, le=120)


class GenerationState(BaseModel):
    state_id: str
    text_before: str
    text_after: str
    input_tokens: list[TokenState]
    visible_context: list[TokenState]
    candidates: list[CandidateState]
    selected: CandidateState
    other_probability_mass: float = Field(ge=0.0, le=1.0)
    temperature: float
    mode: DecodeMode
    entropy: float
    model: ModelInfo
