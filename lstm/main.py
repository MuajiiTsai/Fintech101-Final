from model import finmodel
from dataset import fin_dataset

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

import plotly.express as px
"""
ref: Enhanced credit card fraud detection based on attention mechanism and LSTM deep model
https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00541-8

Steps

Feature Selection: Swarm intelligence algorithm (?) -> manually

Feature Extraction: UMAP (preserve more local/global data structure)
-> using embedding features for training and testing

Data Preprocessing: Synthetic Minority Oversampling Technique (SMOTE)
https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

Model: input -> attention(?) -> LSTM*2 (with dropout) -> Linear -> BN -> SoftMax
"""

#--------------------------------
# TODO
TRAIN_DATA = "../train_clean.csv"
TEST_DATA = "../public_clean.csv"
OUT_PKL = "../output_pkl/RF.pkl"
#--------------------------------

train_df = pd.read_csv(TRAIN_DATA, engine='pyarrow')
train_df = train_df.sort_values(by=['label', 'cano'], ascending=False)[:100]
print("train:", TRAIN_DATA)
test_df = pd.read_csv(TEST_DATA, engine='pyarrow')[:100]
print("test: ", TEST_DATA)

# test_label = pd.DataFrame(np.full(len(test_df), 0.5), columns=['label'])
# test_df = pd.concat((test_df, test_label), axis=1)

# train_df = pd.concat((train_df, test_df), axis=0)

# training
train_key = train_df['txkey'].to_list()
train_label = train_df['label'].to_numpy(dtype=np.float32)
train = train_df.drop(columns=['label', 'cano', 'txkey', 'tid']).to_numpy(dtype=np.float32)

X_train, X_valid, y_train, y_valid = train_test_split(
    train, train_label, test_size=.33, random_state=430
)

# testing
test_key = test_df['txkey'].to_list()
test = test_df.drop(columns=['txkey', 'cano', 'tid']).to_numpy(dtype=np.float32)

# Feature Extraction: TSNE
# embedding = TSNE(n_components=2, 
#                  verbose=True, 
#                  n_iter=250,
#                  n_jobs=-1).fit_transform(train_df.drop(columns=['cano', 'txkey', 'tid', 'label']))
# print(embedding.shape)
embedding = np.ones((len(train_df), 2), dtype=np.float32)

# fig = px.scatter(
#     embedding, x=0, y=1,
#     color=train_df['label'], labels={'color': 'label'}
# )
# fig.show()

#----------------------
# dataset & dataloader
#----------------------
train_dataset = fin_dataset(embedding, train_key, train_label)
train_loader = DataLoader(train_dataset, batch_size=8)
#----------------
# model setting
#----------------
model = finmodel(input_size=2, hidden_size=10)
model = model.cuda()

hidden = (torch.zeros((2, 10), device="cuda"),
          torch.zeros((2, 10), device="cuda"))

criterion=nn.BCELoss()
optimizer=optim.SGD(model.parameters(), lr=0.1)

epochs = 10
for epoch in range(epochs):
    for fin_info, label, txkey in tqdm(train_loader):
        model.train()
        # model.hidden = model.init_hidden()
        optimizer.zero_grad()
        
        fin_info = fin_info.cuda()
        label = label.cuda()
        pred, hidden = model(fin_info, (hidden[0].detach(), hidden[1].detach()))
        
        loss = criterion(torch.flatten(pred), label)
        loss.backward()
        optimizer.step()
    