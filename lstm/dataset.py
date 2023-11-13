import torch
import os
import numpy as np
import ast
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

transform = T.Compose([
    T.ToTensor()
])

class fin_dataset(Dataset):
    '''
    path: ./
    train: if there's no label, train=False.
    '''
    def __init__(self, matrix, txkey, label=None, train=True):
        # datalist = pd.read_csv(path, engine="pyarrow")
        self.matrix = torch.from_numpy(matrix)
        self.train = train
        if train:
            self.train_label = torch.from_numpy(label)
        else:
            self.train_label = None
        self.train_txkey = txkey
                
    def __len__(self):
        return self. matrix.shape[0]

    def __getitem__(self, idx):
        # TODO: done preprocessing before input
        fin_info = self.matrix[idx]
        if self.train:
            label = torch.zeros(2)       
            label[int(self.train_label[idx].item())] = 1  
        else:
            label = torch.nan
        txkey = self.train_txkey[idx]
        
        return fin_info, label, txkey  # return img name
        
if __name__ == '__main__':
    p =  np.ones((100,3))
    label = np.zeros((100,1))
    txkey = ["1" for _ in range(100)]
    d = fin_dataset(p, txkey, label=label, train=True)
    loader = DataLoader(dataset=d, batch_size=8, shuffle=True)
    a, b, c = next(iter(loader))
    # print( type(a), type(b), type(c))
    print(b)

"""
TODO: show the image
"""