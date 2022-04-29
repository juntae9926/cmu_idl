import os
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from letter_list import LETTER_LIST

class LibriSamples(torch.utils.data.Dataset):

    def __init__(self, data_path, letter_list, train=True):

        self.letter_list = letter_list
        self.x_dir = os.path.join(data_path, "mfcc")
        self.x_list = os.listdir(self.x_dir)

        if train == True:
            self.y_dir = os.path.join(data_path, "transcript")
            self.y_list = os.listdir(self.y_dir)

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, ind):
        X = np.load(os.path.join(self.x_dir, self.x_list[ind]), allow_pickle=True) 
        X = torch.Tensor(X)

        if self.y_dir != None:
            Y = np.load(os.path.join(self.y_dir, self.y_list[ind]), allow_pickle=True) 
            Yy = [self.letter_list.index(y) for y in Y]
            Yy = torch.LongTensor(Yy)
            return X, Yy
        else:
            return X
    
    def collate_fn(batch):
        batch_x = [x for x,y in batch]
        batch_x_pad = pad_sequence(batch_x, batch_first=True) # TODO: pad the sequence with pad_sequence (already imported)
        lengths_x = [x.shape[0] for x in batch_x] # TODO: Get original lengths of the sequence before padding
    
        batch_y = [y for x,y in batch]
        batch_y_pad = pad_sequence(batch_y, batch_first=True) # TODO: pad the sequence with pad_sequence (already imported)
        lengths_y = [y.shape[0] for y in batch_y] # TODO: Get original lengths of the sequence before padding

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)

# class LibriSamplesTest(torch.utils.data.Dataset):

#     def __init__(self, data_path, test_order):

#         # TODO
    
#     def __len__(self):
#         # TODO
    
#     def __getitem__(self, ind):
#         # TODO
    
#     def collate_fn(batch):
#         # TODO