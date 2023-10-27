import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import math
import time

from tqdm import tqdm
from torchvision.ops import MLP
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import f1_score

from focalloss import sigmoid_focal_loss
from dataset import fin_dataset

PATH = 'training.csv'

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # dataset setting
    dataset = fin_dataset(PATH, train=True)
    generator = torch.Generator().manual_seed(1)
    train_set, val_set = random_split(dataset, [0.8, 0.2], generator=generator)
    
    train_loader = DataLoader(dataset=train_set, batch_size=4096, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=1024, shuffle=False)
    
    # model
    model = nn.Sequential(
        MLP(in_channels=13, hidden_channels=[100, 1], activation_layer=nn.ReLU),
        nn.Sigmoid()
    )
    
    model.to(device)
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # training setting
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    best_loss, step, early_stop_count = math.inf, 0, 0
    mean_train_losses = []
    mean_valid_losses = []
    
    for epoch in range(args.num_epoch):
        # training
        model.train() # Set your model to train mode.
        loss_record = []
        train_pbar = tqdm(train_loader, position=0, leave=True)
        
        for info, label, _ in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            info, label = info.to(device), label.to(device)   # Move your data to device. 
            pred = torch.flatten(model(info))
            loss = sigmoid_focal_loss(pred, label, reduction="mean")
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{args.num_epoch}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        # lr_scheduler.step()
        mean_train_loss = sum(loss_record)/len(loss_record)
        
        # evaluation
        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for info, label, _ in tqdm(val_loader):
            info, label = info.to(device), label.to(device)
            with torch.no_grad():
                pred = torch.flatten(model(info))
                loss = sigmoid_focal_loss(pred, label, reduction="mean")
                # print(f1_score(pred, label, task="binary"))
                # print(pred, label)
                # raise KeyboardInterrupt
            loss_record.append(loss.item())
            
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
            
        print(f'Epoch [{epoch+1}/{args.num_epoch}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')    
        mean_train_losses.append(mean_train_loss)
        mean_valid_losses.append(mean_valid_loss)
        
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), f"output/best.ckpt") # Save your best model
            print('Saving model with loss {:.10f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1
        
        if early_stop_count >= 10:
            print(f'\nModel is not improving, so we halt the training session at epoch {epoch}.')
            return best_loss
        
        return best_loss
        
parser = argparse.ArgumentParser()
parser.add_argument("--num_epoch", default=1, type=int, help="num of epoch")
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--lr", default=1e-3, type=float)
args = parser.parse_args()

if __name__ == '__main__':
    main(args)


# fillone = np.ones(df_name.shape[0])
# df_name.insert(1, "pred", fillone)
# df_name = df_name.set_index('txkey')
# df_name.to_csv("1.csv")
# print(df_name)