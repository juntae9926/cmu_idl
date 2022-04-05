from unicodedata import bidirectional
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Network(nn.Module):

    def __init__(self, input_dim = 13, hidden_dim = [1024, 512], num_classes = 41): # You can add any extra arguments as you wish

        super(Network, self).__init__()

        self.embedding = nn.Conv1d(input_dim, hidden_dim[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.ReLU = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim[0], hidden_dim[1], bidirectional=True)
        self.classification = nn.Linear(hidden_dim[1] * 2, num_classes)

    def forward(self, x, len_x):

        # embedding
        x = x.permute(0, 2, 1) # Permute [B x T x C] -> [B x C x T]
        out = self.embedding(x)
        out = self.ReLU(out)

        # lstm
        len_x = torch.div(len_x, 2, rounding_mode="floor").long().cpu() 
        out = out.permute(0, 2, 1) # Permute [B x C x T] -> [B x T x C] for input of pack_padded_sequence
        packed_input = pack_padded_sequence(out, len_x, batch_first=True, enforce_sorted=False) # packed_input.data.shape is [82176, 256] [?, C]
        out, (hn, cn) = self.lstm(packed_input) # out.data.shape is [82176, 1024] = [?, C]
        out, lengths  = pad_packed_sequence(out) # [822, 128, 1024] = [?, B, C]

        # classification
        out = self.classification(out) # [822, 128, 41] = [?, B, C]
        out = nn.LogSoftmax(2)(out) 

        return out, lengths