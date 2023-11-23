import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
#-----------------
# TODO
TRAIN_DATA = "train_ver3.csv"
PUBLIC_DATA = "public_ver3.csv"
PRIVATE_DATA = "private_ver3.csv"
MODEL_PATH = "output_pkl"
MODEL_NAME = "RF_ver3.pkl"
OUT_PATH = "output_csv"
OUT_NAME = f"{MODEL_NAME[:-4]}.csv"
OUT_WHOLE = f"{MODEL_NAME[:-4]}_whole.csv"
EMBEDDING_DATA = "embedding.csv"
EXAMPLE = "example.csv"
#------------------
OUT = f"{OUT_PATH}/{OUT_NAME}"
MODEL = f"{MODEL_PATH}/{MODEL_NAME}"
OUT_WHOLE = f"{OUT_PATH}/{OUT_WHOLE}"

example = pd.read_csv(EXAMPLE, engine='pyarrow').set_index('txkey')
print("ex output:", EXAMPLE)
train_df = pd.read_csv(TRAIN_DATA, engine='pyarrow')
print("train:", TRAIN_DATA)
valid_df = pd.read_csv(PUBLIC_DATA, engine='pyarrow').set_index('txkey')
print("valid:", PUBLIC_DATA)
test_df = pd.read_csv(PRIVATE_DATA, engine='pyarrow').set_index('txkey')
print("test:", PRIVATE_DATA)
# print(valid_df.shape, test_df.shape)

valid_df = valid_df[valid_df.index.isin(example.index)]
test_df = test_df[test_df.index.isin(example.index)]
valid_df = valid_df.reset_index()
test_df = test_df.reset_index()

test_df = pd.concat([valid_df.drop(columns=['label']), test_df], axis=0)
print(test_df.shape)

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

test_key = test_df['txkey'].to_frame()

"""embedding
# embedding = pd.read_csv(EMBEDDING_DATA, engine='pyarrow')

# testing dataset
# test_mask = embedding['label'] == 0.5
# test = embedding[test_mask]
# test_key = test['txkey']
# test_arr = test.drop(columns=['label', 'txkey']).to_numpy(dtype=np.float32)
"""

# normalization
scalar = MinMaxScaler().fit(train)

with open(MODEL, 'rb') as file:
    clf = pickle.load(file)

test_arr = test_df.drop(columns=['txkey']).to_numpy(dtype=np.float32)
for i in range(100):
    out = clf.predict(scalar.transform(test_arr)).astype(int)
    print(sum(out))
    test_df['label'] = out
    test_df['llabel'] = test_df.groupby('cid')['label'].shift(1).fillna(-1)
    test_arr = test_df.drop(columns=['txkey', 'label']).to_numpy(dtype=np.float32)

# pred = clf.predict(scalar.transform(valid))
# print(sum(pred))    # 1874
# print(f"\n--------------\n{f1_score(valid_label, pred)}")

out_df = pd.DataFrame(out, columns=['pred'])

output = pd.concat([test_key.reset_index(), out_df.reset_index(drop=True)], axis=1)
output = output.set_index('txkey')
output.to_csv(OUT)

output_whole = pd.concat([test_df.reset_index(), out_df.reset_index(drop=True)], axis=1)
output_whole = output_whole.set_index('txkey')
output_whole.to_csv(OUT_WHOLE)

#TODO: update label iteratively