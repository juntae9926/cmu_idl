import torch
import torch.nn as nn
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
import os

from ctcdecode import CTCBeamDecoder

import warnings
warnings.filterwarnings('ignore')

from phonemes import PHONEME_MAP
from model import Network
from dataset import LibriSamples, LibriSamplesTest
from utils import calculate_levenshtein

def train(model, train_loader, optimizer, criterion, scheduler, scaler, decoder, distance=False):
    model.train()
    
    total_loss = 0
    total_distance = 0

    for i, (x, y, len_x, len_y) in enumerate(train_loader):
        x = x.cuda()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs, out_lengths = model(x, len_x)
            loss = criterion(outputs, y, out_lengths, len_y)
        
        total_loss += float(loss)

        if distance == True:
            distance = calculate_levenshtein(outputs, y, out_lengths, len_y, decoder, PHONEME_MAP=PHONEME_MAP)
            total_distance += distance 
            batch_bar.set_postfix(loss="{:.04f}".format(loss),lr="{:.04f}".format(optimizer.param_groups[0]['lr'], distance="{:.04f}".format(distance)))
        else:
            batch_bar.set_postfix(loss="{:.04f}".format(loss),lr="{:.04f}".format(optimizer.param_groups[0]['lr']))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        batch_bar.update() 

    if distance == True:
        print("Train Loss {:.04f}, Levenshtein distance {:.04f}, Learning rate {:.04f}".format(float(total_loss/len(train_loader)), float(total_distance/len(train_loader)), float(optimizer.param_groups[0]['lr'])))      
    else:
        print("Train Loss {:.04f}, Learning rate {:.04f}".format(float(total_loss/len(train_loader)), float(optimizer.param_groups[0]['lr'])))



def validation(model, val_loader, criterion, decoder):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc='Validation')
    
    total_loss = 0
    total_distance = 0
    for i, (x, y, len_x, len_y) in enumerate(val_loader):
        x = x.cuda()

        with torch.no_grad():
            outputs, out_lengths = model(x, len_x)

        loss = criterion(outputs, y, out_lengths, len_y)
        total_loss += float(loss)

        distance = calculate_levenshtein(outputs, y, out_lengths, len_y, decoder, PHONEME_MAP=PHONEME_MAP)
        total_distance += distance 

        batch_bar.set_postfix(loss="{:.04f}".format(loss),dis="{:.04f}".format(distance))
        batch_bar.update()   

    print("Valid Loss {:.04f}, Levenshtein distance {:04f}".format(    \
        float(total_loss/len(val_loader)), float(total_distance/len(val_loader))))
    batch_bar.close()

    return loss, total_distance/len(val_loader)

def test(test_loader, decoder, device='cuda', model_path = './checkpoint/val_3.449.pth'):
    model = Network().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, leave=False, position=0, desc='Test')

    preds = []
    for x, len_x in test_loader:
        x = x.cuda()
        
        with torch.no_grad():
            outputs, out_lengths = model(x, len_x)
            beam_results, _, _, out_len = decoder.decode(outputs.permute(1, 0, 2), seq_lens=out_lengths)
            pred = "".join(PHONEME_MAP[j] for j in beam_results[0, 0, :out_len[0, 0]])
            preds.append(pred)

        batch_bar.update()   

    result = []
    for idx, pred in enumerate(preds):
        result.append([idx, pred])
    df = pd.DataFrame(result, columns=['id', 'predictions'])
    df.to_csv('./submission/submission_{:.02f}.csv'.format(model_path.split("/")[-1].split(".")[0]), index=False)



def main():
    epochs = 100
    batch_size = 256
    lr = 2e-3
    root = "./hw3p2_student_data/hw3p2_student_data/"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    train_data = LibriSamples(root, 'train')
    val_data = LibriSamples(root, 'dev')
    test_data = LibriSamplesTest(root, 'test_order.csv')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=LibriSamples.collate_fn) 
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=LibriSamples.collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=LibriSamplesTest.collate_fn)

    print("Batch size: ", batch_size)
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

    for data in val_loader:
        x, y, lx, ly = data 
        print(x.shape, y.shape, lx.shape, ly.shape)
        break

    model = Network().to(device)
    print(model)
    # summary(model, x.to(device), lx) # x and lx are from the previous cell

    criterion = nn.CTCLoss() # TODO: What loss do you need for sequence to sequence models? 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # TODO: Adam works well with LSTM (use lr = 2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * batch_size))
    decoder = CTCBeamDecoder(labels=PHONEME_MAP, log_probs_input=True) # TODO: Intialize the CTC beam decoder

    scaler = torch.cuda.amp.GradScaler()

    val_dis = 0
    best_dis = 100

    for epoch in range(epochs):
        print("Epoch {}/{}".format((epoch+1), epochs))

        # Train
        #batch_bar_train = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
        #train(model, train_loader, optimizer, criterion, scheduler, scaler, decoder, batch_bar_train)
        #batch_bar_train.close()
        model.train()
    
        total_loss = 0
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
        for i, (x, y, len_x, len_y) in enumerate(train_loader):
            x = x.cuda()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs, out_lengths = model(x, len_x)
                loss = criterion(outputs, y, out_lengths, len_y)
            
            total_loss += float(loss)

            batch_bar.set_postfix(loss="{:.04f}".format(loss),lr="{:.04f}".format(optimizer.param_groups[0]['lr']))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            batch_bar.update() 
        batch_bar.close()
        print("Train Loss {:.04f}, Learning rate {:.04f}".format(float(total_loss/len(train_loader)), float(optimizer.param_groups[0]['lr'])))


        # Validation
        
        val_loss, val_dis = validation(model, val_loader, criterion, decoder)



        if val_dis < best_dis:
            if not os.path.isdir("./checkpoint"):
                os.mkdir("./checkpoint")
            torch.save(model.state_dict(), './checkpoint/val_{:.02f}.pth'.format(val_dis))

            best_dis = val_dis
            print("--- best model saved ---")
    
    test(test_loader, decoder, device=device)
        

if __name__ == "__main__":
    main()