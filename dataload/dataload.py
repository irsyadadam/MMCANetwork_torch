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

import aeon
import numpy as np


def load_blink_dataset(PATH = "/DATA/"):
    r"""
    1. Load blink dataset
    2. Seperate dimensions into 2 modalities
    3. Return modalities with data wrapper

    RETURNS:
    Tuple (modality1, modality2). For each modality, there is a train and a test
        
    """
    train_x, train_y = aeon.datasets.load_from_tsfile(PATH + "Blink_TRAIN.ts")
    test_x, test_y = aeon.datasets.load_from_tsfile(PATH + "Blink_TEST.ts")

    train_x, train_y = np.array(train_x), np.array(train_y)
    test_x, test_y = np.array(test_x), np.array(test_y)

    # Separate x dimensions into 2 modalities
    m1_train_x = train_x[:, :2, :]
    m2_train_x = train_x[:, 2:, :]
    m1_test_x, m2_test_x = test_x, test_x

    #preserve labels
    m1_test_x = test_x[:, :, :]
    m2_test_x = test_x[:, :, :]
    m1_test_y, m2_test_y = test_y, test_y

    # Return modalities
    return ((m1_train_x, train_y), (m1_test_x, test_y)), ((m2_train_x, train_y), (m2_test_x, test_y))
