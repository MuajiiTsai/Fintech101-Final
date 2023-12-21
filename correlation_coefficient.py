import pandas as pd

#--------------------------------
# TODO
TRAIN_DATA = "train_ctandtime.csv"
TEST_DATA = "public_ctandtime.csv"
OUT_PKL = "output_pkl/RF_newtime.pkl"
EMBEDDING_DATA = "embedding.csv"
#--------------------------------

df = pd.read_csv(TRAIN_DATA, engine='pyarrow')
print(TRAIN_DATA)

print(df.corr())