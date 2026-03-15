from dataclasses import dataclass


@dataclass
class Config:
    embedding_dim: int = 100
    window_size: int = 5
    num_negatives: int = 5
    epochs: int = 5
    initial_lr: float = 0.025
    min_lr: float = 1e-4
    min_count: int = 5
    subsample_t: float = 1e-5
    max_tokens: int | None = None
    log_every: int = 100_000
    seed: int = 42