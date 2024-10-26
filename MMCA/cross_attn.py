
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

class cross_attn_block(torch.nn.module):
    r"""
    Single Block for Cross Attention
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super(cross_attn_block, self).__init__()

    def forward(self, 
                query: Union[torch.Tensor, OptPairTensor] = None, 
                key: Union[torch.Tensor, OptPairTensor] = None, 
                value: Union[torch.Tensor, OptPairTensor] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError



class cross_attn_model(torch.nn.module):
    r"""
    Model for Cross Attention
    """

    def __init__(self, 
                 dim: int, 
                 heads: Optional[int], 
                 dropout: float = 0.0):
        super(cross_attn_model, self).__init__()

    def forward(self, 
                query: Union[torch.Tensor, OptPairTensor] = None, 
                key: Union[torch.Tensor, OptPairTensor] = None, 
                value: Union[torch.Tensor, OptPairTensor] = None, 
                mask: Optional[torch.Tensor] = None,
                layers: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError
