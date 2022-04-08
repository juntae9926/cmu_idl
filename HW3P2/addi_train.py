import torch
import torch.nn as nn
from torchsummaryX import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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


def validation(model, val_loader, criterion, decoder, writer, epoch, device):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc='Validation')
    
    total_loss = 0
    total_distance = 0
    for i, (x, y, len_x, len_y) in enumerate(val_loader):
        x = x.to(device)

        with torch.no_grad():
            outputs, out_lengths = model(x, len_x)

        loss = criterion(outputs, y, out_lengths, len_y)
        writer.add_scaler("val/epoch", loss, epoch)
        total_loss += float(loss)

        distance = calculate_levenshtein(outputs, y, out_lengths, len_y, decoder, PHONEME_MAP=PHONEME_MAP)
        writer.add_scaler("Levenshtein distance", distance, epoch)
        total_distance += distance 

        batch_bar.set_postfix(loss="{:.04f}".format(loss),dis="{:.04f}".format(distance))
        batch_bar.update()  
    batch_bar.close() 
    print("Valid Loss {:.04f}, Levenshtein distance {:04f}".format(    \
        float(total_loss/len(val_loader)), float(total_distance/len(val_loader))))
    
    return loss, total_distance/len(val_loader)


def main():
    epochs = args.epochs
    batch_size = args.batch
    lr = args.lr
    root = "./hw3p2_student_data/hw3p2_student_data/"
    writer = SummaryWriter()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    train_data = LibriSamples(root, 'train')
    val_data = LibriSamples(root, 'dev')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=LibriSamples.collate_fn) 
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=LibriSamples.collate_fn)

    model = Network()
    model.load_state_dict(torch.load("checkpoint/val_9.00.pth"))
    model = model.to(device)
    print(model)

    criterion = nn.CTCLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs * 3))
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, last_epoch=-1)
    decoder = CTCBeamDecoder(labels=PHONEME_MAP, log_probs_input=True)

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
            writer.add_scalar("loss/train", loss, epoch)
            
            total_loss += float(loss)

            batch_bar.set_postfix(loss="{:.04f}".format(loss),lr="{:.04f}".format(optimizer.param_groups[0]['lr']))

            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_bar.update() 
        batch_bar.close()
        print("Train Loss {:.04f}, Learning rate {:.04f}".format(float(total_loss/len(train_loader)), float(optimizer.param_groups[0]['lr'])))


        # Validation
        _, val_dis = validation(model, val_loader, criterion, decoder, writer, epoch, device=args.device)
        writer.close()

        # Model save
        if val_dis < best_dis:
            if not os.path.isdir(args.best_model):
                os.mkdir(args.best_model)
            torch.save(model.state_dict(), '{}/val_{:.02f}.pth'.format(args.best_model, val_dis))

            best_dis = val_dis
            print("--- best model saved ---")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", type=str, help="Select cuda:0 or cuda:1")
    parser.add_argument("--batch", default=64, type=int, help="Batch size")
    parser.add_argument("--best-model", default="./chechpoint", type=str, help="Save best model path")
    parser.add_argument("--epochs", default=20, type=int, help="Total epochs")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--addi", default=False, type=bool, help="additional_training True/False")
    parser.add_argument("--addi-model", type=str, help="load model to additional training")
    
    args = parser.parse_args()
    print(args.device)
    main()