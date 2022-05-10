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
from dataset import LibriSamples, LibriSamplesTest
from utils import *

from letter_list import LETTER_LIST

letter2idx = {"<sos>": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7,
                        "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13, "N": 14, "O": 15,
                        "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20, "U": 21, "V": 22, "W": 23,
                        "X": 24, "Y": 25, "Z": 26, "'": 27, ' ': 28, '<eos>': 29}
idx2letter = {letter2idx[key]: key for key in letter2idx}

def to_str(y, eos=False):
    results = []
    for idx, y_b in enumerate(y):
        chars = []
        for char in y_b:
            char = char.item()
            if char == letter2idx['<eos>']:
                if eos:
                    chars.append('<eos>')
                break
            chars.append(idx2letter[char])
        results.append(''.join(chars))
    return results

def validation(model, val_loader, criterion, args):
    if args.debug:
        model = Seq2Seq(input_dim=13, vocab_size=30, encoder_hidden_dim=512, decoder_hidden_dim=1024, args=args, embed_dim=256, key_value_size=128)
        model.load_state_dict(torch.load("checkpoint/val_3.16.pth"))
        model.to(args.device)

    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc='Validation')
    
    total_loss = 0
    total_distance = 0
    for i, (x, y, len_x, len_y) in enumerate(val_loader):
        x = x.to(args.device)
        y = y.to(args.device)

        with torch.no_grad():
            outputs = model(x, len_x, y, mode='val')

        loss_matrix = criterion(outputs, y)
        loss = torch.sum(loss_matrix) / torch.sum(len_y)
        total_loss += loss.item()

        dist = levenshtein(outputs, y, len_x, len_y, args.batch_size) 
        total_distance += dist

        batch_bar.set_postfix(loss="{:.04f}".format(loss.item()),dis="{:.04f}".format(dist))
        batch_bar.update()  
    batch_bar.close() 
    print("Valid Loss {:.04f}, Levenshtein distance {:04f}".format(    \
        float(total_loss/len(val_loader)), float(total_distance/(len(val_loader)))))
    
    return total_loss/len(val_loader), total_distance/(len(val_loader))

def test(args):

    test_data = LibriSamplesTest(args.data_dir)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=LibriSamplesTest.collate_fn)
    
    model = Seq2Seq(input_dim=13, vocab_size=30, encoder_hidden_dim=256, decoder_hidden_dim=1024, args=args, embed_dim=512, key_value_size=256)
    model.load_state_dict(torch.load(args.test_model))
    model.to(args.device)

    model.eval()
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, leave=False, position=0, desc='Test')
    strings = list()

    for i, (x, len_x) in enumerate(test_loader):
        x = x.to(args.device)
        
        with torch.no_grad():
            output = model(x, len_x, mode='test')
        output = output.log_softmax(2).transpose(1, 2)
        output = output.cpu()

        arg_max = torch.argmax(output, dim=2)
        batch_size = arg_max.size(0)  # TODO

        for i in range(batch_size):  # Loop through each element in the batch

            h_sliced = arg_max[i]
            lend = (h_sliced == 29).nonzero(as_tuple=False)

            if lend.size(0) != 0:
                h_sliced = h_sliced[:lend[0]]
            else:
                h_sliced = h_sliced

            # letter2index, idx2letter = create_dictionaries(LETTER_LIST)
            h_string = "".join(idx2letter[int(j)] for j in h_sliced)
            # h_string = "".join(idx2letter[int(h_sliced)])
            print (h_string)
            strings.append(h_string)
    
    data = np.array(strings)
    with open("submission.csv", "w+") as f:
        f.write("id,predictions\n")
        for i in range(len(data)):
            f.write("{},{}\n".format(i, data[i]))


def main(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    epochs = args.epochs
    letter_list = LETTER_LIST
    vocab_size = len(letter_list)

    root = os.path.join(args.data_dir, "train")
    dev = os.path.join(args.data_dir, "dev")

    train_data = LibriSamples(root, letter_list)
    val_data = LibriSamples(dev, letter_list)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=LibriSamples.collate_fn) 
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=LibriSamples.collate_fn)

    for data in val_loader:
        x, y, lx, ly = data 
        print(x.shape, y.shape, lx.shape, ly.shape)
        break

    model = Seq2Seq(input_dim=13, vocab_size=vocab_size, encoder_hidden_dim=256, decoder_hidden_dim=1024, args=args, embed_dim=512, key_value_size=256)
    if args.addi == True:
            model.load_state_dict(torch.load(args.addi_model))
    model = model.to(device)
    # model =  nn.DataParallel(model).cuda()
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader)), eta_min=1e-5, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=29)

    val_dis = 0
    best_dis = 100

    for epoch in range(epochs):
        print("Epoch {}/{}".format((epoch+1), epochs))
        if not args.debug:
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
                loss = torch.sum(loss_matrix) / torch.sum(len_y)
                total_loss += loss.item()

                # dist = levenshtein(outputs, y, len_x, len_y, args.batch_size)                
                # batch_bar.set_postfix(loss="{:.04f}".format(loss.item()), dist="{}".format(dist), lr="{:.06f}".format(optimizer.param_groups[0]['lr']))

                batch_bar.set_postfix(loss="{:.04f}".format(loss.item()), lr="{:.06f}".format(optimizer.param_groups[0]['lr']))

                loss.backward()
                optimizer.step()
                scheduler.step()

                batch_bar.update() 
            batch_bar.close()
            print("Train Loss {:.04f}, Learning rate {:.06f}".format(float(total_loss/len(train_loader)), float(optimizer.param_groups[0]['lr'])))

            # Validation
            val_loss, val_dis = validation(model, val_loader, criterion, args)

            # Model save
            torch.save(model.state_dict(), 'curr_model_{:02f}.pth'.format(val_dis))
            if val_dis < best_dis:
                if not os.path.isdir(args.save_model):
                    os.mkdir(args.save_model)
                torch.save(model.state_dict(), '{}/val_{:.02f}.pth'.format(args.save_model, val_dis))

                best_dis = val_dis
                print("--- best model saved at {} ---".format(args.save_model))
        
    # Test
    if args.test:
        test(args)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../hw4p2/hw4p2_student_data/hw4p2_student_data", type=str)
    parser.add_argument("--device", default="cuda", type=str, help="Select cuda:0 or cuda:1")
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size")
    parser.add_argument("--save-model", default="./checkpoint", type=str, help="Save best model path")
    parser.add_argument("--epochs", default=100, type=int, help="Total epochs")
    parser.add_argument("--lr", default=5e-4, type=float, help="Learning rate")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--addi-model", type=str, help="load model to additional training")
    parser.add_argument("--test-model", type=str)
    parser.add_argument("--addi", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args)