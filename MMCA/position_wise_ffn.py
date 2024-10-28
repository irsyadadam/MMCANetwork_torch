
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


class position_wise_ffn(torch.nn.Module):
    r"""
    Position-wise feed-forward network with a RELU activation - essentially contracts output, and squeezes it back to the same space

    ARGS:
        dim: dimension of the embeddings
        hidden_dim: dimension of the inflated hidden layer in feed-forward network
    
    """

    def __init__(self, 
                 dim: int, 
                 hidden_dim: int, 
                 dropout: float = 0.0):
        super(position_wise_ffn, self).__init__()

        self.ffn_1 = torch.nn.Linear(dim, hidden_dim)
        self.ffn_2 = torch.nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.ffn_1(x).relu()
        x = self.ffn_2(x)

        return x