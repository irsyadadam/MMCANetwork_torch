
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


class cross_attn_block(torch.nn.Module):
    r"""
    Single Block for Cross Attention

    Args: 
        m1: first modality
        m2: second modality

    Shapes: 
        m1: (seq_length, N_samples, N_features)
        m2: (seq_length, N_samples, N_features)

    Returns: 
        embedding of m1 depending on attending on certain elements of m2, multihead_attn(k_m1, v_m1, q_m2)
    """

    def __init__(self, 
                 dim: int, 
                 heads: int, 
                 dropout: float, 
                 seq_length: int,
                 add_positional: Optional[bool] = False):

        super(cross_attn_block, self).__init__()

        #not learnable, output is x + positional
        self.add_positional = add_positional
        if self.add_positional:
            self.positional_encoding = PositionalEncoding(dim, dropout, seq_length)

        #learnable
        self._to_key = torch.nn.Linear(dim, dim)
        self._to_query = torch.nn.Linear(dim, dim)
        self._to_value = torch.nn.Linear(dim, dim)

        self.attn = torch.nn.MultiheadAttention(embed_dim = dim, num_heads = heads, dropout = dropout)

    def forward(self, 
                m1_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None, 
                m2_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if self.add_positional:
            m1_x = self.positional_encoding(m1_x)
            m2_x = self.positional_encoding(m2_x)

        m1_k = self._to_key(m1_x)
        m1_v = self._to_query(m1_x)
        m2_q = self._to_value(m2_x)

        #crossing
        cross_x, attn_weights = self.attn(m1_k, m1_v, m2_q)

        return cross_x



class cross_attn_channel(torch.nn.Module):
    r"""
    Model for Cross Attention, architecture implementation taken from encoder layer of "Attention is all you need"
    Includes multi-head attn with crossing --> add + norm --> positionwise ffn --> add + norm --> output (based on paper)
    """

    def __init__(self, 
                 dim_m1: int, 
                 dim_m2: int, 
                 pffn_dim: int,
                 heads: Optional[int], 
                 seq_len: int, 
                 dropout: float = 0.0):
        r"""
        ARGS: 
            dim_m1: dim of representations of m1
            dim_m2: dim of representations of m2
            pffn_dim: dim of hidden layer of positional-wise ffn
            heads: number of heads for multi-head attn
            seq_len: length of seq
            dropout: dropout rate

        """
        super(cross_attn_channel, self).__init__()

        self.m1_cross_m2 = cross_attn_block(dim = dim_m1, heads = heads, dropout = dropout, seq_length = seq_len)
        self.m2_cross_m1 = cross_attn_block(dim = dim_m2, heads = heads, dropout = dropout, seq_length = seq_len)

        self.norm_m1 = torch.nn.LayerNorm(dim_m1)
        self.norm_m2 = torch.nn.LayerNorm(dim_m2)

        self.m1_pffn = position_wise_ffn(dim_m1, pffn_dim)
        self.m2_pffn = position_wise_ffn(dim_m2, pffn_dim)

        self.norm_pffn_m1 = torch.nn.LayerNorm(dim_m1)
        self.norm_pffn_m2 = torch.nn.LayerNorm(dim_m2)

        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, 
                m1: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None, 
                m2: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        r"""

        ARGS: 
            m1: input tensor of (seq_length, N_samples, N_features)
            m2: input tensor of (seq_length, N_samples, N_features)
            mask: NOT IMPLEMENTED

        RETURNS:
            tranformed m1, m2, with output dim same as input dim, but with attention
            
        """

        m1_x = self.m1_cross_m2(m1, m2)
        m2_x = self.m2_cross_m1(m2, m1)

        m1_x = self.norm_m1(m1 + self.dropout(m1_x))
        m2_x  = self.norm_m2(m2 + self.dropout(m2_x))

        m1_ffn = self.m1_pffn(m1_x)
        m2_ffn = self.m2_pffn(m2_x)

        m1_x = self.norm_pffn_m1(m1_x + self.dropout(m1_ffn))
        m2_x = self.norm_pffn_m2(m2_x + self.dropout(m2_ffn))

        return m1_x, m2_x

