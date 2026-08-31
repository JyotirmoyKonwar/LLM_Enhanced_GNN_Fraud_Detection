from typing import Iterable

import torch
from torch_geometric.utils import k_hop_subgraph


@torch.no_grad()
def filter_edges(edge_index: torch.Tensor, embeddings: torch.Tensor, limit: int) -> torch.Tensor:
    """Retain each source node's top-k most similar outgoing neighbors."""
    scores = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    scores = scores[edge_index[0]] * scores[edge_index[1]]
    scores = scores.sum(-1)
    keep = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
    for source in edge_index[0].unique():
        candidates = (edge_index[0] == source).nonzero(as_tuple=False).view(-1)
        keep[candidates[torch.topk(scores[candidates], min(limit, candidates.numel())).indices]] = True
    return edge_index[:, keep]


@torch.no_grad()
def semantic_neighbors(
    graph, text_embeddings: torch.Tensor, targets: Iterable[int], hops: int, limit: int
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """Keep the most text-similar neighbors in each target's local subgraph."""
    embeddings = torch.nn.functional.normalize(text_embeddings.float(), dim=-1)
    edge_index = graph.edge_index.cpu()
    result = []
    for target in targets:
        target = int(target)
        nodes, local_edges, _, _ = k_hop_subgraph(target, hops, edge_index, relabel_nodes=True)
        candidates = nodes[nodes != target]
        if candidates.numel() > limit:
            scores = embeddings[candidates] @ embeddings[target]
            candidates = candidates[torch.topk(scores, limit).indices]
            keep = torch.isin(nodes, torch.cat((candidates, nodes.new_tensor([target]))))
            remap = torch.full((nodes.max().item() + 1,), -1, dtype=torch.long)
            remap[nodes[keep]] = torch.arange(keep.sum())
            local_edges = remap[local_edges[:, torch.all(keep[local_edges], dim=0)]]
            nodes = nodes[keep]
        result.append((target, nodes, local_edges))
    return result