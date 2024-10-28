
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
                m2_self_attn_layers: Optional[int] = None,
                m2_self_attn_heads: Optional[int] = None, 
                m2_cross_attn_layers: Optional[int] = None,
                m2_cross_attn_heads: Optional[int] = None,
                #classifier
                classifier: Optional[Any] = None):

        super(MMCA, self).__init__()

        # Multi-modal self-attention layers
        # modality 1 self-attention
        self.m1_self_attn = nn.Sequential(*[
            self_attn_block(m1_shape[1], m1_self_attn_heads, dropout=0.1, seq_length=m1_shape[0]) 
            for _ in range(m1_self_attn_layers)
        ])
        
        # modality 1 cross-attention
        self.m1_cross_attn = nn.Sequential(*[
            cross_attn_block(m1_shape[1], m1_cross_attn_heads, dropout=0.1, seq_length=m1_shape[0]) 
            for _ in range(m1_cross_attn_layers)
        ])
        
        # modality 2 self-attention
        self.m2_self_attn = nn.Sequential(*[
            self_attn_block(m2_shape[1], m2_self_attn_heads, dropout=0.1, seq_length=m2_shape[0]) 
            for _ in range(m2_self_attn_layers)
        ])
        
        # modality 2 cross-attention
        self.m2_cross_attn = nn.Sequential(*[
            cross_attn_block(m2_shape[1], m2_cross_attn_heads, dropout=0.1, seq_length=m2_shape[0]) 
            for _ in range(m2_cross_attn_layers)
        ])
        
        # classification head 
        self.classifier = classifier or nn.Linear(8, 2) # default classifier 

    
    def forward(self, 
            time_series_1: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            time_series_2: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        
        
        #1. Run through self_attn channel
        for layer in self.m1_self_attn:
            m1_self_attn_x = layer(time_series_1)
        
        for layer in self.m2_self_attn:
            m2_self_attn_x = layer(time_series_2)
        
        
        #2. Run through cross_attn channel
        m1_cross_attn_x = time_series_1
        m2_cross_attn_x = time_series_2
        for layer in self.m1_cross_attn:
            m1_cross_attn_x = layer(m1_cross_attn_x, time_series_2)
        
        for layer in self.m2_cross_attn:
            m2_cross_attn_x = layer(m2_cross_attn_x, time_series_1)
        
        
        # print(f"m1_self_attn_x: {m1_self_attn_x.shape}")
        # print(f"m2_self_attn_x: {m2_self_attn_x.shape}")
        # print(f"m1_cross_attn_x: {m1_cross_attn_x.shape}")
        # print(f"m2_cross_attn_x: {m2_cross_attn_x.shape}")
        
        #3 process (?)
        
        #4. Concatenate the outputs from all channels
        # print(f'concatenating channels')
        concatenated_x = torch.cat([m1_self_attn_x, m1_cross_attn_x, m2_self_attn_x, m1_cross_attn_x], dim = -1)
        # print(f'done. concatenated_x has shape {concatenated_x.shape}')
        
        #5. process (?)
        # Olivia: I am simply using mean pooling along the seq dimension to 
        # reduce to a single feature vector. 
        # print(f'average pooling')
        pooled_concat_X = concatenated_x.mean(dim=0)
        # print(f'done. pooled_concat_X has shape {pooled_concat_X.shape}')

        #6. Classify the concatenated output
        x = self.classifier(pooled_concat_X)

        return x, torch.sigmoid(x)




