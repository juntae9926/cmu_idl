import torch
import torch.nn as nn
from torchsummaryX import summary
from torch.utils.data import DataLoader

import pandas as pd
from tqdm import tqdm
import os

from ctcdecode import CTCBeamDecoder
import argparse

import warnings
warnings.filterwarnings('ignore')

from phonemes import PHONEME_MAP
from model import Network
from dataset import LibriSamples, LibriSamplesTest
from utils import calculate_levenshtein


def validation(model, val_loader, criterion, decoder, device):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc='Validation')
    
    total_loss = 0
    total_distance = 0
    for i, (x, y, len_x, len_y) in enumerate(val_loader):
        x = x.to(device)

        with torch.no_grad():
            outputs, out_lengths = model(x, len_x)

        loss = criterion(outputs, y, out_lengths, len_y)
        total_loss += float(loss)

        distance = calculate_levenshtein(outputs, y, out_lengths, len_y, decoder, PHONEME_MAP=PHONEME_MAP)
        total_distance += distance 

        batch_bar.set_postfix(loss="{:.04f}".format(loss),dis="{:.04f}".format(distance))
        batch_bar.update()  
    batch_bar.close() 
    print("Valid Loss {:.04f}, Levenshtein distance {:04f}".format(    \
        float(total_loss/len(val_loader)), float(total_distance/len(val_loader))))
    
    return loss, total_distance/len(val_loader)


def main():
    epochs = 100
    batch_size = args.batch
    lr = 2e-3
    root = "./hw3p2_student_data/hw3p2_student_data/"

    device = args.device if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    train_data = LibriSamples(root, 'train')
    val_data = LibriSamples(root, 'dev')
    test_data = LibriSamplesTest(root, 'test_order.csv')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=LibriSamples.collate_fn) 
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=LibriSamples.collate_fn)

    for data in val_loader:
        x, y, lx, ly = data 
        print(x.shape, y.shape, lx.shape, ly.shape)
        break

    model = Network().to(device)
    print(model)

    criterion = nn.CTCLoss() # TODO: What loss do you need for sequence to sequence models? 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # TODO: Adam works well with LSTM (use lr = 2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * batch_size / 2))
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, last_epoch=-1)
    decoder = CTCBeamDecoder(labels=PHONEME_MAP, log_probs_input=True) # TODO: Intialize the CTC beam decoder

    val_dis = 0
    best_dis = 100

    for epoch in range(epochs):
        print("Epoch {}/{}".format((epoch+1), epochs))

        model.train()
    
        total_loss = 0
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
        for i, (x, y, len_x, len_y) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()


            outputs, out_lengths = model(x, len_x)
            loss = criterion(outputs, y, out_lengths, len_y)
            
            total_loss += float(loss)

            batch_bar.set_postfix(loss="{:.04f}".format(loss),lr="{:.04f}".format(optimizer.param_groups[0]['lr']))

            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_bar.update() 
        batch_bar.close()
        print("Train Loss {:.04f}, Learning rate {:.04f}".format(float(total_loss/len(train_loader)), float(optimizer.param_groups[0]['lr'])))


        # Validation
        val_loss, val_dis = validation(model, val_loader, criterion, decoder, device=args.device)

        # Model save
        if val_dis < best_dis:
            if not os.path.isdir("./checkpoint"):
                os.mkdir("./checkpoint")
            torch.save(model.state_dict(), './checkpoint/val_{:.02f}.pth'.format(val_dis))

            best_dis = val_dis
            print("--- best model saved ---")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", type=str, help="Select cuda:0 or cuda:1")
    parser.add_argument("--batch", default=64, type=int, help="Select batch size")
    parser.add_argument("--best-model", default="./chechpoint", type=str, help="save best model path")

    args = parser.parse_args()
    print(args.device)
    main()