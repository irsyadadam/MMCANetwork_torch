
import importlib.metadata
import json
import logging
import os
import re
import tempfile
import time
import ast
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

import torch


class PositionalEncoding(torch.nn.Module):
    r"""
    Implemented from "Language Modeling with nn.Transformer and TorchText" 

    To inject positional information into the embeddings, we use add a embedding based on the mapping of sin/cosine to our original embedding. 
    REMARKS: do we need to add this if our representations already host positional information?

    Args: 
        d_model: dimension of the embeddings, where embedding is shape [seq_length n_sample, embedding_dim (d_model)]
    """

    def __init__(self, d_model: int, dropout: float = 0.1, seq_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_length n_sample, embedding_dim], same as input for imaging paper``
        """
        #this transformation is for [seq_length n_sample, embedding_dim]
        x = x + self.pe[:x.size(0)]
        
        return self.dropout(x)