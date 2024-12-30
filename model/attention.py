# model/attention.py

import torch
import torch.nn.functional as F
from typing import Tuple

def compute_attention(query: torch.Tensor, 
                     keys: torch.Tensor, 
                     values: torch.Tensor, 
                     scale: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention weights and weighted values.
    
    Args:
        query: Query tensor [batch_size, embed_dim]
        keys: Key tensors [batch_size, num_keys, embed_dim]
        values: Value tensors [batch_size, num_keys, embed_dim]
        scale: Whether to apply scaling to attention scores
        
    Returns:
        Tuple of:
            - attention weights [batch_size, num_keys]
            - weighted sum of values [batch_size, embed_dim]
    """
    # Compute attention scores
    attention = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))
    
    if scale:
        attention = attention / torch.sqrt(torch.tensor(keys.size(-1)))
    
    # Normalize attention weights
    attention_weights = F.softmax(attention, dim=-1)
    
    # Compute weighted sum of values
    output = torch.bmm(attention_weights, values)
    
    return attention_weights.squeeze(1), output.squeeze(1)