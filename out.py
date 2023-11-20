import pickle
import pandas as pd
import numpy as np

#-----------------
# TODO
TEST_DATA = "public_ctandtime.csv"
MODEL_PATH = "output_pkl"
MODEL_NAME = "RF_dif.pkl"
OUT_PATH = "output_csv"
OUT_NAME = f"{MODEL_NAME[:-4]}.csv"
EMBEDDING_DATA = "embedding.csv"
#------------------
OUT = f"{OUT_PATH}/{OUT_NAME}"
MODEL = f"{MODEL_PATH}/{MODEL_NAME}"

test_df = pd.read_csv(TEST_DATA, engine='pyarrow')

test_key = test_df['txkey'].to_frame()
test_arr = test_df.drop(columns=['txkey']).to_numpy(dtype=np.float32)

# embedding = pd.read_csv(EMBEDDING_DATA, engine='pyarrow')

# testing dataset
# test_mask = embedding['label'] == 0.5
# test = embedding[test_mask]
# test_key = test['txkey']
# test_arr = test.drop(columns=['label', 'txkey']).to_numpy(dtype=np.float32)

with open(MODEL, 'rb') as file:
    clf = pickle.load(file)

out = clf.predict(test_arr).astype(int)
print(sum(out))
out = pd.DataFrame(out, columns=['pred'])
output = pd.concat([test_key, out], axis=1)
output = output.set_index('txkey')
output.to_csv(OUT)