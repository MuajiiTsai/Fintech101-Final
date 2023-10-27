import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision.ops import MLP
# from torchmetrics.functional import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import fin_dataset

#TODO: threshold

PATH = 'output/best.ckpt'
PUBLIC_DATA = 'public_processed.csv'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = nn.Sequential(
    MLP(in_channels=13, hidden_channels=[100, 1], activation_layer=nn.ReLU),
    nn.Sigmoid()
)

save_dict = torch.load(PATH)
model.load_state_dict(save_dict)
model.cuda()

dataset = fin_dataset(PUBLIC_DATA, train=False)
loader = DataLoader(dataset, batch_size=20000)

df = pd.DataFrame(columns=["txkey", "pred"])
for info, _, txkey in tqdm(loader):   # no label
    info = info.to(device)
    pred = model(info)
    temp = {
        "txkey": txkey,
        "pred": (torch.flatten(pred).cpu().detach().numpy() > 0.5).astype(np.int32)     # threshold
    }
    temp_df = pd.DataFrame(temp)
    df = pd.concat([df, temp_df], ignore_index=True)
    
df = df.set_index('txkey')
df.to_csv('output_csv/output.csv')

print(df)


