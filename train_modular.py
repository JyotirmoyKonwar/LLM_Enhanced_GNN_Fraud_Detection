import argparse
import random

import numpy as np
import torch

from gemma_gat.config import GemmaGATConfig
from gemma_gat.data import load_graph, split_indices
from gemma_gat.llm import GemmaEnhancer
from gemma_gat.model import GemmaGATModel
from gemma_gat.trainer import GemmaGATTrainer


def main():
    parser = argparse.ArgumentParser(description="Train GemmaGAT with Gemma 3 1B and GAT.")
    parser.add_argument("--dataset", required=True, help="Path such as datasets/reddit.pt")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--llm-name", default="google/gemma-3-1b-it")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--neighbors-per-node", type=int, default=10)
    args = parser.parse_args()
    config = GemmaGATConfig(dataset=args.dataset, output_dir=args.output_dir, llm_name=args.llm_name, epochs=args.epochs, neighbors_per_node=args.neighbors_per_node)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    graph = load_graph(config.dataset)
    enhancer = GemmaEnhancer(config.llm_name, config.device, config.max_input_tokens, config.max_new_tokens, config.lora_rank, config.lora_alpha, config.lora_dropout)
    sample = enhancer.encode([graph.raw_texts[0]])
    model = GemmaGATModel(sample.shape[-1], config.hidden_dim, config.num_classes, config.heads).to(config.device)
    train_indices = split_indices(graph, "train"); val_indices = split_indices(graph, "val")
    GemmaGATTrainer(model, enhancer, graph, config).fit(train_indices, val_indices)


if __name__ == "__main__":
    main()