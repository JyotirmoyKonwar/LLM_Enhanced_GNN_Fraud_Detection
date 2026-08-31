# LLM Enhanced GNN for Fraud Detection

## Gemma 3 1B + GAT

This repository contains a modular implementation of FLAG, the LLM-enhanced graph fraud-detection method. It uses `google/gemma-3-1b-it` for node-text enhancement and `GATConv` exclusively for graph learning. The training objective combines discriminative (causal), residual, and orthogonality losses.

## Project Files

- `datasets/instagram.pt` and `datasets/reddit.pt`: bundled graph datasets.
- `gemma_gat/`: reusable data, Gemma, semantic-sampling, GAT, loss, and trainer modules.
- `train_modular.py`: command-line training entry point.
- `FLAG_Gemma3_GAT_Training.ipynb`: standalone notebook containing the complete implementation. It does not import `gemma_gat`.
- `FLAG/`: original reference implementation.

## Installation

Use a virtual environment with a compatible PyTorch build. For CUDA, install the appropriate PyTorch wheel first from the official PyTorch instructions, then install the remaining dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Gemma 3 1B requires Hugging Face access. Accept the model license and authenticate before loading the model:

```bash
huggingface-cli login
```

The model is loaded in bfloat16 on CUDA and float32 on CPU. CPU execution is possible but generally impractical for full training because Gemma generation and encoding are expensive.

## Notebook

Open `FLAG_Gemma3_GAT_Training.ipynb` in VS Code or Jupyter. In the first code cell, set the dataset and output directory:

```python
DATASET_PATH = Path("datasets/reddit.pt")
OUTPUT_DIR = Path("artifacts/notebook-reddit")
```

Run the cells in order. The final cell performs training and validation, then saves `gat.pt` and the Gemma LoRA adapter under `OUTPUT_DIR`. Use `datasets/instagram.pt` to train on Instagram instead.

## Command Line

Train either bundled graph with the modular implementation:

```bash
python3 train_modular.py --dataset datasets/reddit.pt --output-dir artifacts/reddit
python3 train_modular.py --dataset datasets/instagram.pt --output-dir artifacts/instagram
```

Useful options include `--epochs`, `--neighbors-per-node`, `--llm-name`, and `--output-dir`. The default language model is `google/gemma-3-1b-it`.

## Dataset Contract

Each `.pt` file must load as a PyTorch Geometric `Data` object with:

- `edge_index`: graph edges with shape `[2, num_edges]`.
- `y`: integer node labels.
- `raw_texts`: one text string per node.
- `train_mask` and `val_mask`: boolean node masks.

The implementation expects binary labels (`0` and `1`) and currently evaluates validation AUC and macro-F1. The datasets are loaded from `datasets/`; no `Reddit/` or `Instagram/` directory is required.

## Outputs

For each run, the selected output directory contains:

- `gat.pt`: trained GAT classifier weights.
- `gemma-lora/`: trained Gemma LoRA adapter and configuration.
