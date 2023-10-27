import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

class fin_dataset(Dataset):
    '''
    path: ./
    train: if there's no label, train=False.
    '''
    def __init__(self, path:str, train:bool=True):
        datalist = pd.read_csv(path, engine="pyarrow")
        
        self.train_label = torch.from_numpy(datalist['label'].values).to(torch.float32) if train else None
        self.train_txkey = datalist['txkey'].values
        
        if train:
            self.input = torch.from_numpy(datalist.drop(columns=['label', 'acqic', 'cano', 'chid', 'csmam', 'csmcu', 'etymd', 'hcefg', 'insfg', 'mchno', 'ovrlt', 'scity', 'txkey']).fillna(-1).to_numpy(np.float32))
        else:
            self.input = torch.from_numpy(datalist.drop(columns=['acqic', 'cano', 'chid', 'csmam', 'csmcu', 'etymd', 'hcefg', 'insfg', 'mchno', 'ovrlt', 'scity', 'txkey']).fillna(-1).to_numpy(np.float32))
        
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        # TODO: done preprocessing before input
        fin_info = self.input[idx]        
        label = self.train_label[idx] if self.train_label is not None else torch.nan
        txkey = self.train_txkey[idx]
        
        return fin_info, label, txkey  # return the imgname
        
if __name__ == '__main__':
    p =  "public_processed.csv"
    d = fin_dataset(p, train=False)
    loader = DataLoader(dataset=d, batch_size=8, shuffle=True)
    print(next(iter(loader)))

"""
TODO: show the image
"""