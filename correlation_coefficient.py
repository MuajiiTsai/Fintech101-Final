import pandas as pd

#--------------------------------
# TODO
TRAIN_DATA = "training.csv"
TEST_DATA = "public_ctandtime.csv"
OUT_PKL = "output_pkl/RF_newtime.pkl"
EMBEDDING_DATA = "embedding.csv"
#--------------------------------

df = pd.read_csv(TRAIN_DATA, engine='pyarrow')
print(TRAIN_DATA)

print(df.drop(columns=['acqic', 'chid', \
    'mchno', 'stscd']).corr())