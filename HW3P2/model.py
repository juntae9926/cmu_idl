from unicodedata import bidirectional
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.5)

        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)
        out = self.dropout(out)
        return out

class ResNet(nn.Module):
    def __init__(self, layers, in_channels):
        super(ResNet, self).__init__()
        self.hidden_channels = 1024
        self.conv = nn.Conv1d(in_channels, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(1024)
        self.activation = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size = 3, stride=2, padding=1)

        self.block1 = self.make_layer(1024, layers[0])
        self.block2 = self.make_layer(2048, layers[1])
        #self.block3 = self.make_layer(2048, layers[2])

        self.dropout = nn.Dropout(0.5)
        
    def make_layer(self, out_channels, block_number, stride=1):
        downsample = None
        if (stride != 1) or (self.hidden_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv1d(self.hidden_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        if block_number == 0:
            return nn.Identity()
        layers = []
        layers.append(BasicBlock(self.hidden_channels, out_channels, stride, downsample)) 
        self.hidden_channels = out_channels

        for _ in range(1, block_number):
            layers.append(BasicBlock(self.hidden_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.pool(out)

        out = self.block1(out)
        out = self.block2(out)
        #out = self.block3(out)
        out = self.dropout(out)

        return out

class Network(nn.Module):

    def __init__(self, hidden_dim = [2048, 1024], num_classes = 41): # You can add any extra arguments as you wish
        super(Network, self).__init__()

        self.embedding = ResNet([1, 1], 13)
        self.lstm = nn.LSTM(hidden_dim[0], hidden_dim[1], num_layers=2, bidirectional=True)
        self.activation = nn.GELU()
        # self.classification = nn.Sequential(nn.Linear(hidden_dim[1] * 2, hidden_dim[1]), nn.Linear(hidden_dim[1], num_classes))
        self.classification = nn.Linear(hidden_dim[1] * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, len_x):

        # embedding
        x = x.permute(0, 2, 1) # Permute [B x T x C] -> [B x C x T]
        out = self.embedding(x)
        out = self.dropout(out)

        # lstm
        len_x = torch.div(len_x, 2, rounding_mode="floor").long().cpu() 
        out = out.permute(0, 2, 1) # Permute [B x C x T] -> [B x T x C] for input of pack_padded_sequence
        packed_input = pack_padded_sequence(out, len_x, batch_first=True, enforce_sorted=False) # packed_input.data.shape is [82176, 256] [?, C]
        out, (hn, cn) = self.lstm(packed_input) # out.data.shape is [82176, 1024] = [?, C]
        out, lengths  = pad_packed_sequence(out) # [822, 128, 1024] = [?, B, C]
        out = self.activation(out)

        # classification
        out = self.classification(out) # [822, 128, 41] = [?, B, C]
        out = nn.LogSoftmax(2)(out) 

        return out, lengths