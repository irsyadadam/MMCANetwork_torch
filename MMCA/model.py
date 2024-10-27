
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


class MMCA(torch.nn.Module):
    r"""
    Torch implentation of Multi-Modal Cross Attention
    """
    def __init__(self, 
                #input shapes (Embedding Size, Time Length) 
                m1_shape: Optional[Tuple[int, int]] = None,
                m2_shape: Optional[Tuple[int, int]] = None,
                #modality 1
                m1_self_attn_layers: Optional[int] = None,
                m1_self_attn_heads: Optional[int] = None, 
                m1_cross_attn_layers: Optional[int] = None,
                m1_cross_attn_heads: Optional[int] = None,
                #modality 2
                m1_self_attn_layers: Optional[int] = None,
                m1_self_attn_heads: Optional[int] = None, 
                m1_cross_attn_layers: Optional[int] = None,
                m1_cross_attn_heads: Optional[int] = None,
                #classifier
                classifier: Optional[Any]) -> None:

        super(MMCA, self).__init__()

        # Multi-modal self-attention layers
        #TODO: replace 
        self.m1_self_attn = None 
        self.m1_cross_attn = None

        self.m2_self_attn = None
        self.m2_cross_attn = None

        self.classifier = None

    def forward(self, 
                time_series_1: Union[torch.Tensor, torch.OptPairTensor],
                time_series_2: Union[torch.Tensor, torch.OptPairTensor]) -> torch.Tensor:
        
        
        #1. Run through self_attn channel
        m1_self_attn_x = self.m1_self_attn(time_series_1)
        m2_self_attn_x = self.m2_self_attn(time_series_2)
        
        #2. Run through cross_attn channel
        m1_cross_attn_x = self.m1_cross_attn(time_series_1, time_series_2)
        m2_cross_attn_x = self.m2_cross_attn(time_series_2, time_series_1)
        
        #3 process (?)

        #4. Concatenate the outputs from all channels
        concatenated_x = torch.cat([m1_self_attn_x, m1_cross_attn_x, m2_self_attn_x, m1_cross_attn_x], dim = 1)

        #5. process (?)

        #6. Classify the concatenated output
        x = self.classifier(concatenated_x)

        return x, sigmoid(x)




