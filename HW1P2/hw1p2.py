import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pandas as pd

class Network(torch.nn.Module):
    def __init__(self, n_input=13*41, n_output=40):
        super().__init__()
        # TODO: Please try different architectures
        layers = [
            nn.Linear(n_input, 2048),
            nn.BatchNorm1d(num_features=2048),
              nn.ReLU(),
              #nn.Dropout(0.2),
              nn.Linear(2048, 2048),
              nn.BatchNorm1d(num_features=2048),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(2048, 2048),
              nn.BatchNorm1d(num_features=2048),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(2048, 2048),
              nn.BatchNorm1d(num_features=2048),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(2048, 2048),
              nn.BatchNorm1d(num_features=2048),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(2048, 2048),
              nn.BatchNorm1d(num_features=2048),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(2048, 1024),
              nn.BatchNorm1d(num_features=1024),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(1024, 1024),
              nn.BatchNorm1d(num_features=1024),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(1024,512),
              nn.ReLU(),
              nn.Linear(512, n_output)
        ]

        self.laysers = nn.Sequential(*layers)
        
    def forward(self, A0):
        x = self.laysers(A0)
        return x

class LibriSamples_real(torch.utils.data.Dataset):
    def __init__(self, data_path, sample=20000, partition="test-clean", csvpath=None):
        self.sample = sample
        self.X_dir = data_path + "/" + partition + "/mfcc/"
        self.X_names = os.listdir(self.X_dir)
        
        if csvpath:
            self.X_names = list(pd.read_csv(csvpath).file)
        
        self.length = len(self.X_names)

    def __len__(self):
        return int(np.ceil(self.length / self.sample))
        
    def __getitem__(self, i):
        sample_range = range(i*self.sample, min((i+1)*self.sample, self.length))
        
        X = []
        for j in sample_range:
            X_path = self.X_dir + self.X_names[j]
            X_data = np.load(X_path)
            X_data = (X_data - X_data.mean(axis=0))/X_data.std(axis=0)
            X.append(X_data)
            
        X = np.concatenate(X)
        return X
        
        
class LibriSamples(torch.utils.data.Dataset):
    def __init__(self, data_path, sample=20000, shuffle=True, partition="dev-clean", csvpath=None):
        
        # sample represent how many npy files will be preloaded for one __getitem__ call
        self.sample = sample 

        self.X_dir = data_path + "/" + partition + "/mfcc/"
        self.Y_dir = data_path + "/" + partition +"/transcript/"
        
        self.X_names = os.listdir(self.X_dir)
        self.Y_names = os.listdir(self.Y_dir)

        # using a small part of the dataset to debug
        if csvpath:
            subset = self.parse_csv(csvpath)
            self.X_names = [i for i in self.X_names if i in subset]
            self.Y_names = [i for i in self.Y_names if i in subset]
        
        if shuffle == True:
            XY_names = list(zip(self.X_names, self.Y_names))
            random.shuffle(XY_names)
            self.X_names, self.Y_names = zip(*XY_names)
        
        assert(len(self.X_names) == len(self.Y_names))
        self.length = len(self.X_names)
        
        self.PHONEMES = [
            'SIL',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',  
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '<sos>', '<eos>']
      
    @staticmethod
    def parse_csv(filepath):
        subset = []
        with open(filepath) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                subset.append(row[1])
        return subset[1:]

    def __len__(self):
        return int(np.ceil(self.length / self.sample))
        
    def __getitem__(self, i):
        sample_range = range(i * self.sample, min((i + 1) * self.sample, self.length))
        X, Y = [], []
        for j in sample_range:
            X_path = self.X_dir + self.X_names[j]
            Y_path = self.Y_dir + self.Y_names[j]
 
            label = [self.PHONEMES.index(yy) for yy in np.load(Y_path)][1:-1]
            X_data = np.load(X_path)
            X_data = (X_data - X_data.mean(axis=0)) / X_data.std(axis=0)
            X.append(X_data)
            Y.append(np.array(label))

        X, Y = np.concatenate(X), np.concatenate(Y)
        return X, Y
    
class LibriItems(torch.utils.data.Dataset):
    def __init__(self, X, Y, context = 0):
        assert(X.shape[0] == Y.shape[0])
        
        self.length  = X.shape[0]
        self.context = context

        if context == 0:
            self.X, self.Y = X, Y
        else:
            self.X, self.Y = np.pad(X, ((self.context, self.context), (0, 0)), 'constant', constant_values=0), Y
            pass 
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, i):
        if self.context == 0:
            xx = self.X[i].flatten()
            yy = self.Y[i]
        else:
            xx = self.X[i: self.context*2 + i + 1].flatten()
            yy = self.Y[i]
            pass
        return xx, yy

class LibriItems_real(torch.utils.data.Dataset):
    def __init__(self, X, context = 0):
        
        self.length  = X.shape[0]
        self.context = context

        if context == 0:
            self.X = X
        else: 
            self.X = np.pad(X, ((self.context, self.context), (0, 0)), 'constant', constant_values=0)
            pass 
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, i):
        if self.context == 0:
            xx = self.X[i].flatten()

        else:
            xx = self.X[i: self.context*2 + i + 1].flatten()

        return xx

def train(args, model, device, train_samples, optimizer, criterion, epoch):
    model.train()
    for i in range(len(train_samples)):
        X, Y = train_samples[i]
        train_items = LibriItems(X, Y, context=args['context'])
        train_loader = torch.utils.data.DataLoader(train_items, batch_size=args['batch_size'], shuffle=True, num_workers=2)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float().to(device)
            target = target.long().to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, dev_samples):
    model.eval()
    true_y_list = []
    pred_y_list = []
    with torch.no_grad():
        for i in range(len(dev_samples)):
            X, Y = dev_samples[i]

            test_items = LibriItems(X, Y, context=args['context'])
            test_loader = torch.utils.data.DataLoader(test_items, batch_size=args['batch_size'], num_workers=2, shuffle=False)

            for data, true_y in test_loader:
                data = data.float().to(device)
                true_y = true_y.long().to(device)                
                
                output = model(data)
                pred_y = torch.argmax(output, axis=1)

                pred_y_list.extend(pred_y.tolist())
                true_y_list.extend(true_y.tolist())
        train_accuracy =  accuracy_score(true_y_list, pred_y_list)
    return train_accuracy

def test_real(args, model, device, test_samples):
    model.eval()
    pred_y_list = []
    with torch.no_grad():
        for i in range(len(test_samples)):
            X = test_samples[i]
            test_items = LibriItems_real(X, context=args['context'])
            test_loader = torch.utils.data.DataLoader(test_items, batch_size=args['batch_size'], shuffle=False)
            
            for data in test_loader:
                data = data.float().to(device)
                output = model(data)
                pred_y = torch.argmax(output, axis=1)
                pred_y_list.extend(pred_y.tolist())
    return pred_y_list

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Network().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.9)
    
    # If you want to use full Dataset, please pass None to csvpath
    train_samples = LibriSamples(data_path = args['LIBRI_PATH'], shuffle=True, partition="train-clean-100", csvpath="../input/trainsubset8192/train_filenames_subset_8192_v2.csv")
    full_train_samples = LibriSamples(data_path = args['LIBRI_PATH'], shuffle=True, partition="train-clean-100")
    dev_samples = LibriSamples(data_path = args['LIBRI_PATH'], shuffle=True, partition="dev-clean")
    test_samples = LibriSamples_real(data_path = args['LIBRI_PATH'], partition="test-clean", csvpath="../input/11-785-s22-hw1p2/test_order.csv") 
    
    print("----- Training Start -----")
    for epoch in range(1, args['epoch'] + 1):
        train(args, model, device, full_train_samples, optimizer, criterion, epoch)
        dev_acc = test(args, model, device, dev_samples)
        scheduler.step()
        print('Dev accuracy ', dev_acc)
    
    print("----- Inference Start -----")
    results = test_real(args, model, device, test_samples)    

    print("----- Making csv file -----")
    result = []
    for idx, content in enumerate(results):
        result.append([idx, content])
    df = pd.DataFrame(result, columns=['id', 'label'])
    df.to_csv('submission.csv', index=False)
    
    

if __name__ == '__main__':
    args = {
        'batch_size': 2048*2,
        'context': 20,
        'log_interval': 200,
        'LIBRI_PATH': '/kaggle/input/11-785-s22-hw1p2/hw1p2_student_data',
        'lr': 0.0005,
        'epoch': 20
    }
    main(args)