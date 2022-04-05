import torch

import os
import numpy as np
import pandas as pd
import csv
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from phonemes import PHONEME_MAP, PHONEMES

PHONEMES_MAP = PHONEME_MAP
PHONEMES = PHONEMES


class LibriSamples(torch.utils.data.Dataset):

    def __init__(self, data_path, partition= "train", phonemes=PHONEMES): # You can use partition to specify train or dev

        self.X_dir = data_path + partition + "/mfcc/"
        self.Y_dir = data_path + partition + '/transcript/'

        self.X_files = os.listdir(self.X_dir)
        self.Y_files = os.listdir(self.Y_dir)

        # TODO: store PHONEMES from phonemes.py inside the class. phonemes.py will be downloaded from kaggle.
        # You may wish to store PHONEMES as a class attribute or a global variable as well.
        self.PHONEMES = phonemes

        assert(len(self.X_files) == len(self.Y_files))
        self.length = len(self.X_files)

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, ind):

        X = np.load(self.X_dir + self.X_files[ind]) # TODO: Load the mfcc npy file at the specified index ind in the directory
        Y = np.load(self.Y_dir + self.Y_files[ind]) # TODO: Load the corresponding transcripts
        X = torch.FloatTensor(X)
        
        # Remember, the transcripts are a sequence of phonemes. Eg. np.array(['<sos>', 'B', 'IH', 'K', 'SH', 'AA', '<eos>'])
        # You need to convert these into a sequence of Long tensors
        # Tip: You may need to use self.PHONEMES
        # Remember, PHONEMES or PHONEME_MAP do not have '<sos>' or '<eos>' but the transcripts have them. 
        # You need to remove '<sos>' and '<eos>' from the trancripts. 
        # Inefficient way is to use a for loop for this. Efficient way is to think that '<sos>' occurs at the start and '<eos>' occurs at the end.
        Yy = [self.PHONEMES.index(y) for y in Y[1:-1]]
        
        Yy = torch.LongTensor(Yy) # TODO: Convert sequence of  phonemes into sequence of Long tensors

        return X, Yy
    
    def collate_fn(batch):

        batch_x = [x for x,y in batch]
        batch_y = [y for x,y in batch]

        batch_x_pad = pad_sequence(batch_x, batch_first=True) # TODO: pad the sequence with pad_sequence (already imported)
        lengths_x = [x.shape[0] for x in batch_x] # TODO: Get original lengths of the sequence before padding

        batch_y_pad = pad_sequence(batch_y, batch_first=True) # TODO: pad the sequence with pad_sequence (already imported)
        lengths_y = [y.shape[0] for y in batch_y] # TODO: Get original lengths of the sequence before padding

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)


# You can either try to combine test data in the previous class or write a new Dataset class for test data
class LibriSamplesTest(torch.utils.data.Dataset):

    def __init__(self, data_path, test_order): # test_order is the csv similar to what you used in hw1

        test_order_list = pd.read_csv(os.path.join(data_path, 'test',test_order)).file # TODO: open test_order.csv as a list
        self.X = list(test_order_list) # TODO: Load the npy files from test_order.csv and append into a list
        # You can load the files here or save the paths here and load inside __getitem__ like the previous class
        self.data_path = os.path.join(data_path, 'test/mfcc')

    @staticmethod
    def parse_csv(filepath):
        subset = []
        with open(filepath) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                subset.append(row[0])
        return subset[1:]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, ind):
        # TODOs: Need to return only X because this is the test dataset
        X = np.load(os.path.join(self.data_path, self.X[ind]))
        X = torch.FloatTensor(X)
        return X
    
    def collate_fn(batch):
        batch_x = [x for x in batch]
        batch_x_pad = pad_sequence(batch_x, batch_first=True) # TODO: pad the sequence with pad_sequence (already imported)
        lengths_x = [x.shape[0] for x in batch_x_pad]# TODO: Get original lengths of the sequence before padding

        return batch_x_pad, torch.tensor(lengths_x)