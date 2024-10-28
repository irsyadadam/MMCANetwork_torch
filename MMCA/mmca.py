
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

#import files
from MMCA.cross_attn import cross_attn_channel
from MMCA.self_attn import self_attn_channel
from MMCA.positional_encoder import PositionalEncoding

class multi_modal_cross_attention(torch.nn.Module):
    r"""
    Torch implentation of Multi-Modal Cross Attention

    ARGS:
        *both m1 and m2 have the same shape
        m1_shape: (seq_length, N_samples, N_features)
        m2_shape: (seq_length, N_samples, N_features)

        m1_self_attn_layers: Number of self-attention layers for modality 1
        m1_self_attn_heads: Number of attention heads for modality 1
        m2_self_attn_layers: Number of self-attention layers for modality 2
        m2_self_attn_heads: Number of attention heads for modality 2
        
        cross_attn_layers: Number of cross-attention layers for modality 1
        cross_attn_heads: Number of attention heads for modality 1 cross-attention

        m2_cross_attn_layers: Number of cross-attention layers for modality 2
        m2_cross_attn_heads: Number of attention heads for modality 2 cross-attention

        dropout: dropout applied to all layers
        pffn_dim: hidden dim of position-wise feed-forward layer (optional)
        classifier: classifier for the final prediction (optional)
    """
    def __init__(self, 
                #input shapes (seq_length, N_samples, N_features)
                m1_shape: Optional[Tuple[int, int, int]] = None,
                m2_shape: Optional[Tuple[int, int, int]] = None,
                #modality 1
                m1_self_attn_layers: Optional[int] = None,
                m1_self_attn_heads: Optional[int] = None, 
                #modality 2
                m2_self_attn_layers: Optional[int] = None,
                m2_self_attn_heads: Optional[int] = None, 
                #cross
                cross_attn_heads: Optional[int] = None,
                cross_attn_layers: Optional[int] = None,
                #classifier
                dropout: Optional[int] = 0.1,
                pffn_dim: Optional[int] = None,
                add_positional: Optional[bool] = False,
                classifier: Optional[Any] = None):

        super(multi_modal_cross_attention, self).__init__()

        self.add_positional = add_positional
        if add_positional:
            self._to_positional_m1 = PositionalEncoding(m1_shape[-1], dropout, m1_shape[0])
            self._to_positional_m2 = PositionalEncoding(m2_shape[-1], dropout, m2_shape[0])

        pffn_dim = pffn_dim or 200
        self.m1_self_attn = torch.nn.Sequential(*[
            self_attn_channel(dim = m1_shape[-1], pffn_dim = pffn_dim, heads = m1_self_attn_heads, seq_len = m1_shape[0], dropout = dropout)
            for _ in range(m1_self_attn_layers)
        ])

        self.m2_self_attn = torch.nn.Sequential(*[
            self_attn_channel(dim = m2_shape[-1], pffn_dim = pffn_dim, heads = m2_self_attn_heads, seq_len = m2_shape[0], dropout = dropout)
            for _ in range(m2_self_attn_layers)
        ])
        
        # cross_attention channel --> takes in both sequences, returns both
        #TODO: might need to change to "cross_attn_block" if we want to do a single cross
        self.mm_cross_attn = torch.nn.Sequential(*[
            cross_attn_channel(dim_m1 = m1_shape[-1], dim_m2 = m2_shape[-1], pffn_dim = pffn_dim, heads = cross_attn_heads, seq_len = m2_shape[0], dropout = dropout)
            for _ in range(cross_attn_layers)
        ])
        

        #TODO: Change final classifier to fit imaging data
        self.classifier = classifier or torch.nn.Linear(8, 2) # default classifier 

    
    def forward(self, 
            time_series_1: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            time_series_2: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        
        #0. Add poitional
        if self.add_positional:
            time_series_1 = self._to_positional_m1(time_series_1)
            time_series_2 = self._to_positional_m2(time_series_2)

        #1. Run through self_attn channel
        for layer in self.m1_self_attn:
            m1_self_attn_x = layer(time_series_1)
        
        for layer in self.m2_self_attn:
            m2_self_attn_x = layer(time_series_2)
        
        
        #2. Run through cross_attn channel
        for layer in self.mm_cross_attn:
            m1_cross_attn_x, m2_cross_attn_x = layer(time_series_1, time_series_2)
        
        #3 process (?)
        
        #4. Concatenate the outputs from all channels
        # print(f'concatenating channels')
        concatenated_x = torch.cat([m1_self_attn_x, m1_cross_attn_x, m2_self_attn_x, m1_cross_attn_x], dim = -1)
        # print(f'done. concatenated_x has shape {concatenated_x.shape}')
        
        #5. avg pooling
        pooled_concat_X = concatenated_x.mean(dim=0)

        #6. Classify the concatenated output
        x = self.classifier(pooled_concat_X)

        return x, torch.softmax(x, dim = 1)




