import pandas as pd
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor
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
def process_group(group):
    difCityTime = [0]  # Initialize the list with 0
    print(group['cid'][0])
    lastCity = group['ctdif'] != 0.0
    tdif = group['tdif'].tolist()
    last = 0

    for i in range(1, len(lastCity)):
        if last == 0 and not lastCity.iloc[i]:
            pre = difCityTime[-1]
            difCityTime.append(tdif[i] + pre)
        else:
            difCityTime.append(tdif[i])

        last = lastCity.iloc[i]
    
    group['tdifCity'] = difCityTime

    # print(group[['cid', 'tdif', 'ctdif', 'tdifCity']])
    return group
# data
DATA1 = "training.csv"
DATA2 = "public.csv"
DATA3 = "private_1_processed.csv"

d1 = pd.read_csv(DATA1, engine="pyarrow").drop(columns=['acqic', 'chid', 'contp', 'insfg', 'iterm', 'bnsfg', 'flbmk'\
    , 'mchno', 'stscd'])
print("dataset:", DATA1)
d2 = pd.read_csv(DATA2, engine='pyarrow').drop(columns=['acqic', 'chid', 'contp', 'insfg', 'iterm', 'bnsfg', 'flbmk'\
    , 'mchno', 'stscd'])
print("dataset:", DATA2)
d3 = pd.read_csv(DATA3, engine='pyarrow').drop(columns=['acqic', 'chid', 'contp', 'insfg', 'iterm', 'bnsfg', 'flbmk'\
    , 'mchno', 'stscd'])
print("dataset:", DATA3)

print(d2.shape, d3.shape)

d1['id'] = 1
d2['id'] = 0
d3['id'] = -1

# d1 = d1[:10000]
# d2 = d2[:1000]
# d3 = d3[:1000]

d1_fill = fill(d1.drop(columns=['cano', 'txkey']))
d2_fill = fill(d2.drop(columns=['cano', 'txkey']))
d3_fill = fill(d3.drop(columns=['cano', 'txkey']))

d1 = pd.concat([d1['txkey'], d1['cano'], d1_fill], axis=1)
d2 = pd.concat([d2['txkey'], d2['cano'], d2_fill], axis=1)
d3 = pd.concat([d3['txkey'], d3['cano'], d3_fill], axis=1)

d1_index = d1.set_index('txkey').index
d2_index = d2.set_index('txkey').index
d3_index = d3.set_index('txkey').index

datalist = pd.concat([d1, d2, d3], ignore_index=True)   # label private dataset: -10

datalist = datalist.sort_values(by=['locdt', 'loctm'], ascending=True, kind='stable')   # sorting by date and time
datalist = datalist.sort_values(by=['cano'], ascending=True, kind='stable')
datalist['cid'] = datalist['cano'].rank(method='dense', ascending=True).astype(int)

datalist = datalist.drop(columns=['cano'])

# merge locdt and loctm to 'time'
# Merge 'locdt' and 'loctm' into a new 'time' column
datalist['time'] = pd.to_datetime(datalist['loctm'].astype(int).astype(str).str.zfill(6), format='%H%M%S', errors='coerce')

# Calculate the time difference and store in 'tiff' column
datalist['tdif'] = (datalist.groupby('cid')['time'].diff().dt.total_seconds() + datalist.groupby('cid')['locdt'].diff() * 86400).fillna(-1) 

# Drop the unnecessary columns
datalist = datalist.drop(['locdt', 'loctm', 'time'], axis=1)
# print(datalist[['cid', 'tdif', 'time', 'locdt', 'loctm']])

"""
# time difference (obsolete)
# Sort the DataFrame by CustomerID and Time
datalist = datalist.sort_values(['cid', 'time'])
# Calculate the time difference for the same customer ID
datalist['tdif'] = datalist.groupby('cid')['time'].diff().fillna(-1)
"""

# last city
datalist['lcity'] = datalist.groupby('cid')['stocn'].shift(1).fillna(-1)
# last label
datalist['llabel'] = datalist.groupby('cid')['label'].shift(1).fillna(-1)

# city dif
datalist['ctdif'] = (datalist.groupby('cid')['stocn'].diff().fillna(0).ne(0)).astype(float)

# csmam dif
datalist['amdif'] = abs(datalist.groupby('cid')['flam1'].diff().fillna(0)).astype(float)

datalist = datalist.set_index('txkey')

new_datalist=pd.DataFrame(columns=datalist.columns+'tdifCity')
with ProcessPoolExecutor() as executor:
    # Group by 'cid' column
    grouped_datalist = datalist.groupby('cid')

    #interate over each cid
    for group_key in grouped_datalist.groups.keys():
        print(group_key, "/", datalist['cid'].nunique())
        group=grouped_datalist.get_group(group_key)

        #list of 'tdifCity'
        difCityTime=[] 

        lastCity=(group['ctdif'] != 0.0).tolist()
        tdif=group['tdif'].tolist()
        last=0
        for i in range(len(lastCity)):
            if i ==0:
                if lastCity[i]==0: difCityTime.append(0)
                else :difCityTime.append(tdif[i])
                last=lastCity[i]
            else:
                #print(last,lastCity[i],tdif[i],difCityTime[-1])
                if last==0 and lastCity[i]==0: 
                    pre=difCityTime[-1]
                    difCityTime.append(tdif[i]+pre)
                else: 
                    difCityTime.append(tdif[i])
                last=lastCity[i]
        group['tdifCity']=difCityTime
        new_datalist = pd.concat([new_datalist, group])
datalist=new_datalist
# private output
private_dataset = datalist[datalist.index.isin(d3_index)]
private_dataset = private_dataset.drop(columns=['label'])
# print(private_dataset[:100])
print(private_dataset.shape)
private_dataset.to_csv("private_ver2.csv")
# public output
public_dataset = datalist[datalist.index.isin(d2_index)]
print(public_dataset.shape)
public_dataset.to_csv("public_ver2.csv")
# train output
train_dataset = datalist[datalist.index.isin(d1_index)]
print(train_dataset.shape)
train_dataset.to_csv("train_ver2.csv")
