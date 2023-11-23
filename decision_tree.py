from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os

#--------------------------------
# TODO
TRAIN_DATA = "train_ver3.csv"
PUBLIC_DATA = "public_ver3.csv"
PRIVATE_DATA = "private_ver3.csv"
OUT_PKL = "output_pkl/RF_ver3.pkl"
# EMBEDDING_DATA = "embedding.csv"
#--------------------------------
"""
############
embedding
############

embedding = pd.read_csv(EMBEDDING_DATA, engine='pyarrow')

# training dataset
train_mask = embedding['label'] != 0.5
train = embedding[train_mask]
label = train['label'].to_numpy(dtype=np.float32)
train = train.drop(columns=['label', 'txkey'])

# data split
X_train, X_valid, y_train, y_valid = train_test_split(train, label, stratify=label, test_size=.2)
# train_key, valid_key = X_train['txkey'].tolist(), X_test['txkey'].tolist()
train_label, valid_label = y_train, y_valid
train_arr, valid_arr = X_train.to_numpy(dtype=np.float32), X_valid.to_numpy(dtype=np.float32)
# train_arr, valid_arr = X_train.drop(columns=['txkey']).to_numpy(dtype=np.float32), X_test.drop(columns=['txkey']).to_numpy(dtype=np.float32)

# # testing dataset
# test_mask = embedding['label'] == 0.5
# test = embedding[test_mask]
# test_key = test['txkey'].tolist()
# test_arr = test.drop(columns=['label', 'txkey']).to_numpy(dtype=np.float32)
"""

train_df = pd.read_csv(TRAIN_DATA, engine='pyarrow')
print("train:", TRAIN_DATA)
valid_df = pd.read_csv(PUBLIC_DATA, engine='pyarrow')
print("valid:", PUBLIC_DATA)

# test_df = pd.read_csv(TEST_DATA, engine='pyarrow')
# print("test: ", TEST_DATA)

# training

# excluding the data with only one transaction and a label of 0
# Calculate each person's transaction times
train_df['transaction_times'] = train_df.groupby('cid')['cid'].transform('count')
# Delete rows where label is 0 and there's only one transaction
train_df = train_df[~((train_df['label'] == 0) & (train_df['transaction_times'] == 1))]
# Drop the temporary 'transaction_times' column
train_df = train_df.drop('transaction_times', axis=1)

train_key = train_df['txkey'].to_frame()

label = train_df['label'].to_numpy(dtype=np.float32)
train = train_df.drop(columns=['label', 'txkey']).to_numpy(dtype=np.float32)

valid_label = valid_df['label'].to_numpy(dtype=np.float32)
valid = valid_df.drop(columns=['label', 'txkey']).to_numpy(dtype=np.float32)

X_train, X_valid, y_train, y_valid = train_test_split(
    train, label, test_size=.33, stratify=label
)

# testing
# test_key = test_df['txkey'].to_frame()
# test = test_df.drop(columns=['txkey', 'cano', 'tid']).to_numpy(dtype=np.float32)

# normalization
scalar = MinMaxScaler().fit(train)

clf = RandomForestClassifier(n_estimators=100, max_depth=30, verbose=100, n_jobs=-1)
# clf = MLPClassifier(random_state=1, max_iter=100, verbose=True)
# clf = SVC(gamma=2, C=1, random_state=1, max_iter=250, verbose=True)
# clf = clf.fit(scalar.transform(X_train), y_train)

###### using all training data
clf = clf.fit(scalar.transform(train), label)

pred = clf.predict(scalar.transform(valid))
print(f"\n--------------\n{f1_score(valid_label, pred)}")

# Save the model to a file using pickle
with open(OUT_PKL, 'wb') as file:
    pickle.dump(clf, file)



