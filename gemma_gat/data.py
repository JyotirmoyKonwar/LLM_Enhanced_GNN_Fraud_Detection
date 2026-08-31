from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data


def load_graph(path: Path) -> Data:
    """Load one bundled graph and normalize its text field."""
    try:
        graph: Any = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        graph = torch.load(path, map_location="cpu")
    if not hasattr(graph, "edge_index") or not hasattr(graph, "y"):
        raise ValueError(f"{path} must contain edge_index and y")
    texts = getattr(graph, "raw_texts", None)
    if texts is None:
        raise ValueError(f"{path} must contain raw_texts for Gemma enhancement")
    graph.raw_texts = [str(text) for text in texts]
    graph.y = graph.y.long().view(-1)
    graph.edge_index = graph.edge_index.long()
    return graph


def split_indices(graph: Data, split: str) -> torch.Tensor:
    mask = getattr(graph, f"{split}_mask", None)
    if mask is not None:
        return mask.nonzero(as_tuple=False).view(-1)
    raise ValueError(
        f"Dataset has no {split}_mask. Existing sampler files can be adapted "
        "to this interface by exposing their central node indices."
    )