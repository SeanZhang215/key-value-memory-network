# Key-Value Memory Network for Question Answering

A PyTorch implementation of Key-Value Memory Networks for question answering on biographical data, based on the paper [Key-Value Memory Networks for Directly Reading Documents](https://arxiv.org/abs/1606.03126).

## Overview
This project implements a single-hop attention-based memory network for answering questions about biographical information. The model learns to:
- Match questions against relevant keys in a knowledge base
- Use attention mechanisms to retrieve relevant information
- Return appropriate answers from stored values

## Setup
```bash
git clone https://github.com/SeanZhang215/key-value-memory-network.git
cd key-value-memory-network
pip install -r requirements.txt
```

## Usage
Check `example.ipynb` for a complete walkthrough of:
- Data preprocessing
- Model training
- Question answering inference

Basic usage:
```python
from model.memory_network import KVMemoryNetwork

# Initialize model
model = KVMemoryNetwork(vocab_size=1000, embed_dim=128)

# Train and use model (see example.ipynb for details)
```

## Project Structure
```
key-value-memory-network/
├── model/
│   ├── memory_network.py   # Core model implementation
│   └── attention.py        # Attention mechanism
├── utils/
│   └── data_utils.py       # Data processing utilities
├── train.py                # Training script
└── example.ipynb          # Usage example notebook
```

## Requirements
See requirements.txt for dependencies.

## Reference
Miller, A., et al. "Key-Value Memory Networks for Directly Reading Documents." arXiv:1606.03126, 2016.