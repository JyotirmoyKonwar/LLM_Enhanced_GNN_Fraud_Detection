from pathlib import Path

import torch
from sklearn.metrics import f1_score, roc_auc_score

from .sampling import filter_edges


class GemmaGATTrainer:
    def __init__(self, model, enhancer, graph, config):
        self.model, self.enhancer, self.graph, self.config = model, enhancer, graph, config
        self.llm_optimizer = torch.optim.AdamW(enhancer.model.parameters(), lr=config.llm_lr)
        self.gat_optimizer = torch.optim.AdamW(model.parameters(), lr=config.gat_lr, weight_decay=config.weight_decay)

    def _induced_edges(self, indices):
        indices = indices.cpu().long()
        node_map = torch.full((self.graph.num_nodes,), -1, dtype=torch.long)
        node_map[indices] = torch.arange(indices.numel())
        edges = self.graph.edge_index.cpu()
        keep = (node_map[edges[0]] >= 0) & (node_map[edges[1]] >= 0)
        return node_map[edges[:, keep]].to(self.config.device)

    def run_split(self, indices, train: bool):
        pairs = self.enhancer.generate([self.graph.raw_texts[int(i)] for i in indices])
        causal = self.enhancer.encode([pair.causal for pair in pairs])
        residual = self.enhancer.encode([pair.residual for pair in pairs])
        edge_index = self._induced_edges(indices)
        with torch.no_grad():
            source_embeddings = self.enhancer.encode([self.graph.raw_texts[int(i)] for i in indices])
        edge_index = filter_edges(edge_index, source_embeddings.to(self.config.device), self.config.neighbors_per_node)
        labels = self.graph.y[indices].to(self.config.device)
        causal = causal.to(self.config.device)
        residual = residual.to(self.config.device)
        if train:
            self.model.train(); self.enhancer.model.train(); self.llm_optimizer.zero_grad(); self.gat_optimizer.zero_grad()
        else:
            self.model.eval(); self.enhancer.model.eval()
        with torch.set_grad_enabled(train):
            loss = self.model.loss(causal, residual, edge_index, labels, self.config.alpha, self.config.beta)
            if train:
                loss.backward(); self.llm_optimizer.step(); self.gat_optimizer.step()
        with torch.no_grad():
            logits = self.model.predict(causal, edge_index)
            prediction = logits.argmax(-1).cpu().numpy()
            probability = logits.softmax(-1)[:, 1].cpu().numpy()
        return {"loss": float(loss.detach().cpu()), "f1": f1_score(labels.cpu(), prediction, average="macro"), "auc": roc_auc_score(labels.cpu(), probability) if len(set(labels.cpu().tolist())) > 1 else float("nan")}

    def fit(self, train_indices, val_indices):
        best = float("-inf")
        for epoch in range(self.config.epochs):
            train_metrics = self.run_split(train_indices, True)
            val_metrics = self.run_split(val_indices, False)
            score = val_metrics["f1"] + val_metrics["auc"]
            print(f"epoch={epoch + 1} train={train_metrics} val={val_metrics}")
            if score > best:
                best = score
                Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), Path(self.config.output_dir) / "gat.pt")
                self.enhancer.model.save_pretrained(Path(self.config.output_dir) / "gemma-lora")