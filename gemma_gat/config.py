from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class GemmaGATConfig:
    dataset: Path
    output_dir: Path = Path("artifacts")
    llm_name: str = "google/gemma-3-1b-it"
    hidden_dim: int = 128
    num_classes: int = 2
    heads: int = 4
    neighbor_hops: int = 1
    neighbors_per_node: int = 10
    max_input_tokens: int = 2048
    max_new_tokens: int = 64
    llm_lr: float = 1e-4
    gat_lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 3
    inner_epochs: int = 10
    alpha: float = 0.1
    beta: float = 0.1
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
