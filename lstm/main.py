from model import finmodel
from dataset import fin_dataset
from focalloss import sigmoid_focal_loss

import logging
import math
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from tqdm import tqdm

"""
ref: Enhanced credit card fraud detection based on attention mechanism and LSTM deep model
https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00541-8

Steps:
Feature Selection: Swarm intelligence algorithm (?) -> manually

Feature Extraction: UMAP (preserve more local/global data structure)
-> using embedding features for training and testing

Data Preprocessing: Synthetic Minority Oversampling Technique (SMOTE)
https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

Model: input -> attention(?) -> LSTM*2 (with dropout) -> Linear -> BN -> SoftMax
"""

#--------------------------------
# TODO
TRAIN_DATA = "./train_clean.csv"
TEST_DATA = "./public_clean.csv"
EMBEDDING_DATA = "./embedding.csv"

OUT_HIDDEN = "./output_ckpt/hidden.pt"
OUT_PKL = "./output_pkl/RF.pkl"
CKPT_PATH = "./output_ckpt"

EMBED_DIM = 3
HIDDEN_SIZE = 50
#--------------------------------

embedding = pd.read_csv(EMBEDDING_DATA, engine='pyarrow')

# training dataset
train_mask = embedding['label'] != 0.5
train = embedding[train_mask]
label = train['label'].to_numpy(dtype=np.float32)
train = train.drop(columns=['label'])

# data split
X_train, X_test, y_train, y_test = train_test_split(train, label, stratify=label, test_size=.2)
train_key, valid_key = X_train['txkey'].tolist(), X_test['txkey'].tolist()
train_label, valid_label = y_train, y_test
train_arr, valid_arr = X_train.drop(columns=['txkey']).to_numpy(dtype=np.float32), X_test.drop(columns=['txkey']).to_numpy(dtype=np.float32)

# testing dataset
test_mask = embedding['label'] == 0.5
test = embedding[test_mask]
test_key = test['txkey'].tolist()
test_arr = test.drop(columns=['label', 'txkey']).to_numpy(dtype=np.float32)

# # training
# train_key = train_df['txkey'].to_list()
# train_label = train_df['label'].to_numpy(dtype=np.float32)
# train = train_df.drop(columns=['label', 'cano', 'txkey', 'tid']).to_numpy(dtype=np.float32)

# # testing
# test_key = test_df['txkey'].to_list()
# test = test_df.drop(columns=['txkey', 'cano', 'tid']).to_numpy(dtype=np.float32)

#----------------------
# dataset & dataloader
#----------------------

train_dataset = fin_dataset(train_arr, train_key, train_label)
train_loader = DataLoader(train_dataset, batch_size=4096)
valid_dataset = fin_dataset(valid_arr, valid_key, valid_label)
valid_loader = DataLoader(valid_dataset, batch_size=4096)

test_dataset = fin_dataset(test_arr, test_key, train=False)
test_loader = DataLoader(test_dataset, batch_size=1024)

#----------------
# model setting
#----------------
model = finmodel(input_size=EMBED_DIM, hidden_size=HIDDEN_SIZE)
model = model.cuda()
print("# of model params: ",sum(p.numel() for p in model.parameters() if p.requires_grad))

# initial hidden messages p.s. shape = (num_layers, hidden_size)
hidden = (torch.zeros((2, HIDDEN_SIZE), device="cuda"),
          torch.zeros((2, HIDDEN_SIZE), device="cuda"))

criterion=nn.BCELoss(reduction='mean')
optimizer=optim.SGD(model.parameters(), lr=1e-3)

epochs = 100
best_loss, early_stop_count = math.inf, 0
for epoch in range(epochs):
    print(f"epoch {epoch}:")
    # train
    model.train()
    loss_record = []
    for fin_info, label, txkey in tqdm(train_loader):
        optimizer.zero_grad()
        fin_info = fin_info.cuda()
        label = label.cuda()
        pred, hidden = model(fin_info, (hidden[0].detach(), hidden[1].detach()))
        loss = sigmoid_focal_loss(pred, label, reduction='mean', alpha=0.01, gamma=2)
        # loss = criterion(torch.flatten(pred), label)
        loss.backward()
        optimizer.step()
        loss_record.append(loss.detach().item())
    mean_train_loss = sum(loss_record)/len(loss_record) 
    
    # evaluation
    model.eval()
    loss_record = []
    for fin_info, label, txkey in tqdm(valid_loader):
        fin_info = fin_info.cuda()
        label = label.cuda()
        with torch.no_grad():
            pred, hidden = model(fin_info, (hidden[0].detach(), hidden[1].detach()))
            loss = sigmoid_focal_loss(pred, label, reduction='mean', alpha=0.01, gamma=2)
            # loss = criterion(torch.flatten(pred), label)
        loss_record.append(loss.item())
    mean_valid_loss = sum(loss_record)/len(loss_record)
    logging.info(f"mean train: {mean_train_loss}, ,mean valid: {mean_valid_loss}")
    
    if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), f"{CKPT_PATH}/best.ckpt") # Save your best model
            torch.save(hidden, OUT_HIDDEN)
            print('Saving model with loss {:.10f}...'.format(best_loss))
            early_stop_count = 0
    else: 
        early_stop_count += 1
    
    if early_stop_count == 20:
        print("The model is not improving, so the training session has been halted.")
        break

