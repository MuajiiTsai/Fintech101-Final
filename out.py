import pickle
import pandas as pd
import numpy as np
import time

from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
#-----------------
# TODO
TRAIN_DATA = "train_ver2.csv"
PUBLIC_DATA = "public_ver2.csv"
PRIVATE_DATA = "private_ver2.csv"
MODEL_PATH = "output_pkl"
MODEL_NAME = "RF_allin.pkl"
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

#------------------
# function
#------------------
# Function to predict labels for a group
def predict_labels(group_df, scalar):
    group_indices = group_df.index
    for idx in range(len(group_indices)):
        if group_df.drop(columns=['pred']).loc[group_indices[idx], ['llabel']].values[0] == -1 and idx != 0:
            group_df.at[group_indices[idx], 'llabel'] = group_df.at[group_indices[idx-1], 'pred']
            
        row_features = group_df.drop(columns=['pred']).loc[group_indices[idx]].values.reshape(1, -1)
        predicted_label = clf.predict(scalar.transform(row_features))[0]
        group_df.at[group_indices[idx], 'pred'] = predicted_label
            
    return group_df


valid_df = valid_df[valid_df.index.isin(example.index)].drop(columns=['label'])
test_df = test_df[test_df.index.isin(example.index)]
valid_df = valid_df.reset_index()
test_df = test_df.reset_index()

# test_df = pd.concat([valid_df, test_df], axis=0)
test_df = test_df.set_index('txkey')

# excluding the data with only one transaction and a label of 0
# Calculate each person's transaction times
train_df['transaction_times'] = train_df.groupby('cid')['cid'].transform('count')
# Delete rows where label is 0 and there's only one transaction
train_df = train_df[~((train_df['label'] == 0) & (train_df['transaction_times'] == 1))]
# Drop the temporary 'transaction_times' column
train_df = train_df.drop('transaction_times', axis=1)
train = train_df.drop(columns=['label', 'txkey']).to_numpy(dtype=np.float32)
# normalization
scalar = MinMaxScaler().fit(train)
print(train.shape, test_df.shape)

with open(MODEL, 'rb') as file:
    clf = pickle.load(file)

valid_key = valid_df['txkey'].to_frame()
valid_arr = valid_df.drop(columns=['txkey']).to_numpy(dtype=np.float32)
valid_arr = scalar.transform(valid_arr)
out = clf.predict(valid_arr).astype(int)

out = pd.DataFrame(out, columns=['pred'])
out_valid = pd.concat([valid_key, out], axis=1).set_index('txkey')
out_valid_whole = pd.concat([valid_df, out], axis=1).set_index('txkey')
# output : out_valid, out_valid_whole

# test_key = test_df['txkey'].to_frame()
# test_arr = test_df.drop(columns=['txkey']).to_numpy(dtype=np.float32)
# test_arr = scalar.transform(test_arr)
# out = clf.predict(test_arr).astype(int)

# Group by 'group' column and predict labels in parallel
num_cores = 10  # Set the number of cores you want to use
groups = test_df.groupby('cid')
test_df['pred'] = np.nan
start = time.time()
# Use Parallel to apply the function predict_labels to each group in parallel
results = Parallel(n_jobs=num_cores)(delayed(predict_labels)(group_df, scalar) for name, group_df in groups)

# Concatenate the results into a single DataFrame
out_test_whole = pd.concat(results)
end = time.time()
print(end-start)
# Display the final DataFrame
print(sum(out_test_whole['pred'].values))

out_test = out_test_whole['pred'].to_frame()
out = pd.concat((out_valid, out_test), axis=0)
out_whole = pd.concat((out_valid_whole, out_test_whole), axis=0)
print(out_whole)

out.to_csv(OUT)
out_whole.to_csv(OUT_WHOLE)

#TODO: update label iteratively 