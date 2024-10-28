
import importlib.metadata
import json
import logging
import os
import re
import tempfile
import time
import ast
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

import torch

#import file
from MMCA.positional_encoder import PositionalEncoding
from MMCA.position_wise_ffn import position_wise_ffn


class self_attn_block(torch.nn.Module):
    r"""
    self attention block 

    Args: 
        dim: Dimension of the embeddings for this modality 
        heads: Number of attention heads
        dropout: Dropout rate
        seq_length: Sequence length

    Shapes: 
        x: (seq_length, N_samples, N_features)

    Returns: 
        Embedding of x after self-attention with same input dimensions
    """

    def __init__(self, 
                 dim: int, 
                 heads: int, 
                 dropout: float, 
                 seq_length: int, 
                  add_positional: Optional[bool] = False):

        super(self_attn_block, self).__init__()

        self.add_positional = add_positional
        if self.add_positional:
            self.positional_encoding = PositionalEncoding(dim, dropout, seq_length)

        # learnable linear projections 
        self._to_key = torch.nn.Linear(dim, dim)
        self._to_query = torch.nn.Linear(dim, dim)
        self._to_value = torch.nn.Linear(dim, dim)

        # Multi-head attention layer
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)

    def forward(self, 
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # (optional) positional encoding
        # print('positional encoding...')
        if self.add_positional:
            x = self.positional_encoding(x)
        # print('done!')

        # project input to q, k, v 
        # print('k, q, v')
        k = self._to_key(x)
        q = self._to_query(x)
        v = self._to_value(x)
        # print('done!')

        # Self-attention: each element attends to all elements within the sequence
        # print('self attn')
        attn_x, attn_weights = self.attn(q, k, v, attn_mask=mask)
        # print('done!')

        return attn_x

class self_attn_channel(torch.nn.Module):
    r"""
    Self-Attention Channel Model, based on the architecture of "Attention is All You Need"
    Includes self-attention, add + norm, position-wise FFN, add + norm

    Args:
        dim: Dimension of the embeddings
        pffn_dim: Dimension of hidden layer in position-wise FFN
        heads: Number of attention heads
        seq_len: Length of sequence
        dropout: Dropout rate

    Shapes:
        x: (seq_length, N_samples, N_features)

    Returns:
        Transformed x, same dimension as input with self-attention applied
    """

    def __init__(self, 
                 dim: int, 
                 pffn_dim: int, 
                 heads: Optional[int], 
                 seq_len: int, 
                 dropout: float = 0.0):
        
        super(self_attn_channel, self).__init__()

        # Self-attention block
        self.self_attn = self_attn_block(dim=dim, heads=heads, dropout=dropout, seq_length=seq_len)

        # Layer normalization for self-attention output
        self.norm_self_attn = torch.nn.LayerNorm(dim)

        # Position-wise feed-forward network
        self.pffn = position_wise_ffn(dim, pffn_dim)

        # Layer normalization for FFN output
        self.norm_pffn = torch.nn.LayerNorm(dim)

        # Dropout
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, 
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # self-attn and add residual connection
        attn_x = self.self_attn(x, mask=mask)
        x = self.norm_self_attn(x + self.dropout(attn_x))

        # position-wise ffn and add residual connection
        ffn_x = self.pffn(x)
        x = self.norm_pffn(x + self.dropout(ffn_x))

        return x
