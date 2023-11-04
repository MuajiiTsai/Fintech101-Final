import pandas as pd
import numpy as np
import argparse

"""
length of 
    training.csv: 8688526 
    public: 551407
    private: 
        
"""


DATA1 = "training.csv"
DATA2 = "public_processed.csv"

d1 = pd.read_csv(DATA1, engine="pyarrow").fillna(-100).drop(columns=['acqic', 'bnsfg', 'chid', 'csmam', 'contp', 'csmcu', 'ecfg',\
    'etymd', 'flbmk', 'flam1', 'flg_3dsmk', 'hcefg', 'insfg', 'iterm', 'mchno', 'ovrlt', 'scity', 'stscd'])
print("dataset:", DATA1)
d2 = pd.read_csv(DATA2, engine='pyarrow').fillna(-100).drop(columns=['acqic', 'bnsfg', 'chid', 'csmam', 'contp', 'csmcu', 'ecfg',\
    'etymd', 'flbmk', 'flam1', 'flg_3dsmk', 'hcefg', 'insfg', 'iterm', 'mchno', 'ovrlt', 'scity', 'stscd'])
print("dataset: ", DATA2)

datalist = pd.concat([d1, d2], ignore_index=True).fillna(-10)   # label public dataset: -10
# print(datalist)

datalist = datalist
datalist = datalist.sort_values(by=['locdt', 'loctm'], ascending=True)   # sorting by date and time
tid = np.arange(len(datalist))  # transaction id
datalist.insert(0, "tid", tid)
datalist = datalist.sort_values(by=['cano'], ascending=False, kind='stable')
datalist['cid'] = datalist['cano'].rank(method='dense', ascending=True).astype(int)

corr = datalist[['txkey', 'tid', 'cano', 'cid']]
corr.to_csv("correspondence_table.csv", index=False)

datalist = datalist.drop(columns=['cano', 'txkey'])

# public output
public_mask = datalist['label'] == -10
public_dataset = datalist[public_mask]
public_dataset = public_dataset.drop(columns=['label']) # drop the label
public_dataset = public_dataset.set_index('tid')
public_dataset.to_csv("public_clean.csv")

# train output
train_mask = datalist['label'] != -10
train_dataset = datalist[train_mask]
train_dataset = train_dataset.set_index('tid')
train_dataset.to_csv("train_clean.csv")

# train_mask = datalist[(datalist['label'] != -1) & (df['label'] != -2)] if private dataset is added