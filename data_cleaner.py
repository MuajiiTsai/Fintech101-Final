import pandas as pd
import numpy as np
import argparse

"""
length of 
    training.csv: 8688526 
    public: 551407
    private: 
"""

def fill(df: pd.DataFrame):
    for key in df.keys():
        num_missing_data = df[key].isnull().sum()
        if num_missing_data == 0:
            continue
        if num_missing_data > len(df[key]) // 2: 
            df[key] = df[key].fillna(0)
        else:
            df[key] = df[key].fillna(df[key].mode().values[0])  # filling missing data with the mode
    return df

DATA1 = "training.csv"
DATA2 = "public_processed.csv"

d1 = pd.read_csv(DATA1, engine="pyarrow").drop(columns=['acqic', 'chid', 'conam', 'csmam', \
    'mchno', 'scity', 'stscd'])
print("dataset:", DATA1)
d2 = pd.read_csv(DATA2, engine='pyarrow').drop(columns=['acqic', 'chid', 'conam', 'csmam', \
    'mchno', 'scity', 'stscd'])
print("dataset:", DATA2)

d1_fill = fill(d1.drop(columns=['cano', 'txkey']))
d2_fill = fill(d2.drop(columns=['cano', 'txkey']))
d1 = pd.concat([d1['txkey'], d1['cano'], d1_fill], axis=1)
d2 = pd.concat([d2['txkey'], d2['cano'], d2_fill], axis=1)

datalist = pd.concat([d1, d2], ignore_index=True).fillna(0.5)   # label public dataset: -10

# raise KeyboardInterrupt

# datalist = datalist
datalist = datalist.sort_values(by=['locdt', 'loctm'], ascending=True)   # sorting by date and time
datalist = datalist.sort_values(by=['cano'], ascending=False, kind='stable')
datalist['cid'] = datalist['cano'].rank(method='dense', ascending=True).astype(int)

datalist = datalist.drop(columns=['cano'])

# public output
public_mask = datalist['label'] == 0.5
public_dataset = datalist[public_mask]
public_dataset = public_dataset.drop(columns=['label']) # drop the label
public_dataset = public_dataset.set_index('txkey')
public_dataset = public_dataset.sort_values(by=['cid'], ascending=True)
public_dataset.to_csv("public_clean.csv")

# train output
train_mask = datalist['label'] != 0.5
train_dataset = datalist[train_mask]
train_dataset = train_dataset.set_index('txkey')
train_dataset = train_dataset.sort_values(by=['cid'], ascending=True)
train_dataset.to_csv("train_clean.csv")

# train_mask = datalist[(datalist['label'] != -1) & (df['label'] != -2)] if private dataset is added