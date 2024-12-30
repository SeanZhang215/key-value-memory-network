# model/memory_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class KVMemoryNetwork(nn.Module):
    """
    Key-Value Memory Network implementation for question answering.
    
    This model implements a single-hop attention mechanism over a key-value store,
    where keys are matched against a question embedding to retrieve relevant values.
    
    Attributes:
        vocab_size (int): Size of the vocabulary
        embed_dim (int): Dimension of the embedding space
        A (nn.Linear): Primary embedding layer for questions and keys
        B (nn.Linear): Secondary embedding layer for values
    """
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super(KVMemoryNetwork, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Embedding layers
        self.A = nn.Linear(vocab_size, embed_dim)
        self.B = nn.Linear(vocab_size, embed_dim)
    
    def forward(self, question: torch.Tensor, 
                keys: torch.Tensor, 
                values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the memory network.
        
        Args:
            question: Batch of questions [batch_size, vocab_size]
            keys: Batch of keys [batch_size, num_keys, vocab_size]
            values: Batch of values [batch_size, num_keys, vocab_size]
            
        Returns:
            torch.Tensor: Retrieved memory output [batch_size, embed_dim]
        """
        # Embed question and memory
        question_embedding = self.A(question)
        key_embeddings = self.A(keys)
        value_embeddings = self.A(values)
        
        # Compute attention weights
        question_embedding = question_embedding.unsqueeze(1)
        attention = torch.bmm(question_embedding, key_embeddings.transpose(1, 2))
        attention = F.softmax(attention, dim=-1)
        
        # Weight and sum values
        output = torch.bmm(attention, value_embeddings)
        return output.squeeze(1)
    
    def get_value_embeddings(self, values: torch.Tensor) -> torch.Tensor:
        """Embed values using the B embedding layer."""
        return self.B(values)