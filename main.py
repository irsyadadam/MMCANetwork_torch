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
import math
import aeon

import torch
import torch.nn as nn

import os
import numpy as np
import aeon
from aeon.datasets import load_from_tsfile



if __name__ == '__main__':

    DATA_PATH = "DATA/"

    train_x, train_y = aeon.datasets.load_from_tsfile(DATA_PATH + "Blink_TRAIN.ts")
    test_x, test_y = aeon.datasets.load_from_tsfile(DATA_PATH + "Blink_TEST.ts")

    train_x, train_y = np.array(train_x), np.array(train_y)
    test_x, test_y = np.array(test_x), np.array(test_y)

    #reshape from (sample, feat_dim, seq_length) to (seq_length, sample, feat_dim)
    train_x, test_x = np.transpose(train_x, (2, 0, 1)), np.transpose(test_x, (2, 0, 1))

    #encode the labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_y, test_y = le.fit_transform(train_y), le.fit_transform(test_y)

    # Separate x dimensions into 2 modalities
    m1_train_x = train_x[:, :, :2]
    m2_train_x = train_x[:, :, 2:]
    m1_train_y, m2_train_y = train_y, train_y

    #preserve labels
    m1_test_x = test_x[:, :, :2]
    m2_test_x = test_x[:, :, 2:]
    m1_test_y, m2_test_y = test_y, test_y

    print("--------------DATALOAD Complete----------------\n")

    device = "cuda:0"

    from MMCA.mmca import multi_modal_cross_attention

    mmca_model = multi_modal_cross_attention(
                                m1_shape = m1_train_x.shape, 
                                m2_shape = m2_train_x.shape, 
                                m1_self_attn_layers = 2,
                                m1_self_attn_heads = 2, 
                                m2_self_attn_layers = 2,
                                m2_self_attn_heads = 2,
                                cross_attn_layers = 2,
                                cross_attn_heads = 2,
                                dropout = 0.1,
                                add_positional = True                            
                                )

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(mmca_model.parameters(), 0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.001, 100, 50)

    mmca_model = mmca_model.to(device)

    m1_train_x = torch.Tensor(m1_train_x).to(device)
    m2_train_x = torch.Tensor(m2_train_x).to(device)
    train_y = torch.Tensor(train_y).to(device)

    m1_test_x = torch.Tensor(m1_test_x).to(device)
    m2_test_x = torch.Tensor(m2_test_x).to(device)
    test_y = torch.Tensor(test_y).to(device)


    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, recall_score

    def acc_metrics(probs, preds, y_true):

        accuracy = accuracy_score(y_true = y_true, y_pred = preds)
        precision = precision_score(y_true = y_true, y_pred = preds, average ='macro')
        recall = recall_score(y_true = y_true, y_pred = preds, average ='macro')
        f1 = f1_score(y_true = y_true, y_pred = preds, average ='macro')
        auc = roc_auc_score(y_true = y_true, y_score = probs, average ='macro')

        print(f"Accuracy: {accuracy: .2f}, AUC: {auc: .2f}, Precision: {precision: .2f}, Recall: {recall: .2f}, F1: {f1: .2f}")

    epoch_num = 100

    print("--------------Train Step----------------\n \n")

    for i in range(epoch_num):
        mmca_model.train()
        logits, probs = mmca_model(m1_train_x, m2_train_x)
        loss = criterion(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f'Epoch [{i+1}/{epoch_num}], Loss: {loss.item()}')
        preds = torch.max(probs, 1)

        acc_metrics(probs.detach().cpu().numpy(), preds.detach().cpu().numpy(), train_y.detach().cpu().numpy())
    
    