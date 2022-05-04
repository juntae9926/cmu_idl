import torch
import torch.nn as nn
from torchsummaryX import summary
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tqdm import tqdm
import os

from ctcdecode import CTCBeamDecoder
import argparse

import warnings

from model import pBLSTM
warnings.filterwarnings('ignore')

from model import Seq2Seq
from dataset import LibriSamples
from utils import *

from letter_list import LETTER_LIST

def validation(model, val_loader, criterion, device):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc='Validation')
    
    total_loss = 0
    total_distance = 0
    for i, (x, y, len_x, len_y) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            outputs = model(x, len_x, y)

            loss_matrix = criterion(outputs, y)
            B, T = loss_matrix.shape

            mask = Variable(torch.zeros(B, T), requires_grad=False).to(device)
            for i in range(B):
                l = len_y[i]
                mask[i, :l] = 1
            
            masked_loss = loss_matrix * mask
            loss = torch.sum(masked_loss) / torch.sum(len_y)
            total_loss += loss.item()

        distance = levenshtein(outputs, y)
        total_distance += distance 

        batch_bar.set_postfix(loss="{:.04f}".format(loss.item()),dis="{:.04f}".format(distance))
        batch_bar.update()  
    batch_bar.close() 
    print("Valid Loss {:.04f}, Levenshtein distance {:04f}".format(    \
        float(total_loss/len(val_loader)), float(total_distance/len(val_loader))))
    
    return total_loss/len(val_loader), total_distance/len(val_loader)


def main():
    device = args.device if torch.cuda.is_available() else 'cpu'
    epochs = args.epochs
    letter_list = LETTER_LIST
    vocab_size = len(letter_list)

    root = "./hw4p2_student_data/hw4p2_student_data/train"
    dev = "./hw4p2_student_data/hw4p2_student_data/dev"
    #root = "./hw4p2_simple/hw4p2_simple/train"
    #dev = "./hw4p2_simple/hw4p2_simple/dev"

    train_data = LibriSamples(root, letter_list)
    val_data = LibriSamples(dev, letter_list)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=LibriSamples.collate_fn) 
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=1, collate_fn=LibriSamples.collate_fn)

    for data in val_loader:
        x, y, lx, ly = data 
        print(x.shape, y.shape, lx.shape, ly.shape)
        break

    #model = Network()
    model = Seq2Seq(input_dim=13, vocab_size=vocab_size, encoder_hidden_dim=512, decoder_hidden_dim=512, args=args, embed_dim=256, key_value_size=128)
    if args.addi == True:
            model.load_state_dict(torch.load(args.addi_model))
    model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs))
    criterion = nn.CrossEntropyLoss(reduction='none')

    val_dis = 0
    best_dis = 100

    for epoch in range(epochs):
        print("Epoch {}/{}".format((epoch+1), epochs))

        model.train()
    
        total_loss = 0
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
        for i, (x, y, len_x, len_y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            len_y = len_y.to(device)
            optimizer.zero_grad()

            outputs = model(x, len_x, y)
            
            loss_matrix = criterion(outputs, y)
            B, T = loss_matrix.shape

            mask = Variable(torch.zeros(B, T), requires_grad=False).to(device)
            for i in range(B):
                l = len_y[i]
                mask[i, :l] = 1
            
            masked_loss = loss_matrix * mask
            loss = torch.sum(masked_loss) / torch.sum(len_y)
            total_loss += loss.item()

            batch_bar.set_postfix(loss="{:.04f}".format(loss.item()),lr="{:.04f}".format(optimizer.param_groups[0]['lr']))

            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_bar.update() 
        batch_bar.close()
        print("Train Loss {:.04f}, Learning rate {:.04f}".format(float(total_loss/len(train_loader)), float(optimizer.param_groups[0]['lr'])))

        # Validation
        _, val_dis = validation(model, val_loader, criterion, device=args.device)

        # Model save
        if val_dis < best_dis:
            if not os.path.isdir(args.save_model):
                os.mkdir(args.save_model)
            torch.save(model.state_dict(), '{}/val_{:.02f}.pth'.format(args.save_model, val_dis))

            best_dis = val_dis
            print("--- best model saved at {} ---".format(args.save_model))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1", type=str, help="Select cuda:0 or cuda:1")
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size")
    parser.add_argument("--save-model", default="./chechpoint", type=str, help="Save best model path")
    parser.add_argument("--epochs", default=20, type=int, help="Total epochs")
    parser.add_argument("--lr", default=2e-3, type=float, help="Learning rate")
    parser.add_argument("--addi", default=False, type=bool, help="additional_training True/False")
    parser.add_argument("--addi-model", type=str, help="load model to additional training")

    args = parser.parse_args()
    main()