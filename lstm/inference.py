import torch
from model import finmodel
from dataset import fin_dataset

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

#-------------------------------------------
EMBED_DIM = 3
HIDDEN_SIZE = 50
EMBEDDING_DATA = "./embedding.csv"
HIDDEN = "./output_ckpt/hidden.pt"
checkpoint_path = './output_ckpt/best.ckpt'
#-------------------------------------------

#-------------------------------------------
# load model
#-------------------------------------------

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)
# print(checkpoint)
# Access the model state_dict or other components
model_state_dict = checkpoint
# ... (access other components as needed)

# Create an instance of your model
# (make sure your model architecture matches the one saved in the checkpoint)
model = finmodel(input_size=EMBED_DIM, hidden_size=HIDDEN_SIZE)
model.load_state_dict(model_state_dict)
model = model.cuda()
# Now your_model is loaded with the weights from the checkpoint

#-------------------------------------------
# load data
#-------------------------------------------

embedding = pd.read_csv(EMBEDDING_DATA, engine='pyarrow')

# testing dataset
test_mask = embedding['label'] == 0.5
test = embedding[test_mask]
test_key = test['txkey'].tolist()
test_arr = test.drop(columns=['label', 'txkey']).to_numpy(dtype=np.float32)

test_dataset = fin_dataset(test_arr, test_key, train=False)
test_loader = DataLoader(test_dataset, batch_size=1024)

hidden = torch.load(HIDDEN)

for fin_info, _, txkey in tqdm(test_loader):
    with torch.no_grad():
        fin_info = fin_info.cuda()
        label, hidden = model(fin_info, (hidden[0].detach(), hidden[1].detach()))
        # print("label:", label.shape)
        print(label)
