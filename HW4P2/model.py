from unicodedata import bidirectional
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence
from torch.autograd import Variable

from letter_list import LETTER_LIST

class LockedDropout(nn.Module):
    def __init__(self, dim=1):
        super(LockedDropout,self).__init__()
        self.dim = dim

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5):

        if self.dim == 0:
            mask = torch.zeros((1, x.shape[1], x.shape[2]), requires_grad=False, device=x.device).bernoulli_(1-dropout)
        else:
            mask = torch.zeros((x.shape[0], 1, x.shape[2]), requires_grad=False, device=x.device).bernoulli_(1-dropout)

        return mask * x

class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    Read paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed
    2. Truncate the input length dimension by concatenating feature dimension
        (i) How should  you deal with odd/even length input? 
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_dim * 2, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = LockedDropout(dropout)

    def forward(self, x:PackedSequence):

        x, lengths = pad_packed_sequence(x, batch_first=True)
        B, T, C = x.shape

        # odd length
        if T % 2 == 1:
            x = x[:, :-1, :]
        
        lengths = lengths // 2
        x = x.contiguous().view(B, T//2, C*2)
        x = self.dropout(x)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.blstm(x)[0]

        return x


class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key, value and unpacked_x_len.

    '''
    def __init__(self, input_dim, encoder_hidden_dim, dropout=0.2, key_value_size=128):
        super(Encoder, self).__init__()
        # The first LSTM layer at the bottom

        self.lstm = nn.LSTM(input_dim, encoder_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        # Define the blocks of pBLSTMs
        # Dimensions should be chosen carefully
        # Hint: Bidirectionality, truncation...
        self.pBLSTMs = nn.Sequential(pBLSTM(encoder_hidden_dim * 2, encoder_hidden_dim),
                                    pBLSTM(encoder_hidden_dim * 2, encoder_hidden_dim),
                                    pBLSTM(encoder_hidden_dim * 2, encoder_hidden_dim))
         
        # The linear transformations for producing Key and Value for attention
        # Hint: Dimensions when bidirectional lstm? 
        self.key_network = nn.Linear(encoder_hidden_dim * 2, key_value_size)
        self.value_network = nn.Linear(encoder_hidden_dim * 2, key_value_size)

        self.dropout = LockedDropout(dropout)

    def forward(self, x, x_len):
        """
        1. Pack your input and pass it through the first LSTM layer (no truncation)
        2. Pass it through the pyramidal LSTM layer
        3. Pad your input back to (B, T, *) or (T, B, *) shape
        4. Output Key, Value, and truncated input lens

        Key and value could be
            (i) Concatenated hidden vectors from all time steps (key == value).
            (ii) Linear projections of the output from the last pBLSTM network.
                If you choose this way, you can use the final output of
                your pBLSTM network.
        """


        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x = self.lstm(x)[0]
        x = self.pBLSTMs(x)
        x, lengths = pad_packed_sequence(x, batch_first=True)

        k = self.key_network(x)
        v = self.value_network(x)

        return k, v, lengths

def plot_attention(attention):
    # utility function for debugging
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.show()

class Dot_Attention(nn.Module):
    '''
    Attention is calculated using key and value from encoder and query from decoder.
    Here are different ways to compute attention and context:
    1. Dot-product attention
        energy = bmm(key, query) 
        # Optional: Scaled dot-product by normalizing with sqrt key dimension
        # Check "attention is all you need" Section 3.2.1
    * 1st way is what most TAs are comfortable with, but if you want to explore...
    2. Cosine attention
        energy = cosine(query, key) # almost the same as dot-product xD 
    3. Bi-linear attention
        W = Linear transformation (learnable parameter): d_k -> d_q
        energy = bmm(key @ W, query)
    4. Multi-layer perceptron
        # Check "Neural Machine Translation and Sequence-to-sequence Models: A Tutorial" Section 8.4
    
    After obtaining unnormalized attention weights (energy), compute and return attention and context, i.e.,
    energy = mask(energy) # mask out padded elements with big negative number (e.g. -1e9)
    attention = softmax(energy)
    context = bmm(attention, value)

    5. Multi-Head Attention
        # Check "attention is all you need" Section 3.2.2
        h = Number of heads
        W_Q, W_K, W_V: Weight matrix for Q, K, V (h of them in total)
        W_O: d_v -> d_v

        Reshape K: (B, T, d_k)
        to (B, T, h, d_k // h) and transpose to (B, h, T, d_k // h)
        Reshape V: (B, T, d_v)
        to (B, T, h, d_v // h) and transpose to (B, h, T, d_v // h)
        Reshape Q: (B, d_q)
        to (B, h, d_q // h)

        energy = Q @ K^T
        energy = mask(energy)
        attention = softmax(energy)
        multi_head = attention @ V
        multi_head = multi_head reshaped to (B, d_v)
        context = multi_head @ W_O
    '''
    def __init__(self):
        super(Dot_Attention, self).__init__()
        # Optional: dropout
        #self.linear = nn.Linear(512, 128)

    def forward(self, query, key, value, mask):
        """
        input:
            key: (batch_size, seq_len, d_k)
            value: (batch_size, seq_len, d_v)
            query: (batch_size, d_q)
        * Hint: d_k == d_v == d_q is often true if you use linear projections
        return:
            context: (batch_size, key_val_dim)
        
        """
        # return context, attention
        # we return attention weights for plotting (for debugging)
        #query = self.linear(query)
        query = torch.unsqueeze(query, -1) # [B, L, D]
        energy = torch.bmm(key, query).squeeze(-1)
        energy = torch.masked_fill(energy, mask, -1e9)
        attention = torch.softmax(energy, dim=1).unsqueeze(1)
        context = torch.bmm(attention, value).squeeze(1)

        return context, attention.squeeze(1)


class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step.
    Thus we use LSTMCell instead of LSTM here.
    The output from the last LSTMCell can be used as a query for calculating attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    # key-value-size : attention dimension, en
    def __init__(self, vocab_size, decoder_hidden_dim, args, embed_dim=256, key_value_size=128):
        super(Decoder, self).__init__()
        # Hint: Be careful with the padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # The number of cells is defined based on the paper
        self.lstm1 = nn.LSTMCell(input_size=embed_dim, hidden_size=decoder_hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=decoder_hidden_dim, hidden_size=key_value_size)
    
        self.attention = Dot_Attention()     
        self.vocab_size = vocab_size
        # Optional: Weight-tying
        self.character_prob = nn.Linear(embed_dim, vocab_size)# fill this out) #: d_v -> vocab_size
        self.key_value_size = key_value_size
        
        # Weight tying
        self.character_prob.weight = self.embedding.weight
        self.args = args

    def _forward_step(self, input, context, hidden_states, key, value, mask):

        embed = self.embedding(input)
        h1, c1 = self.lstm1(embed)
        #h2, c2 = self.lstm2(h1, hidden_states[1])
        h2, c2 = self.lstm2(h1)

        if key is not None:
            context, attention = self.attention(h2, key, value, mask)
        else:
            attention = None
        
        return context, [h1, h2], attention

    def forward(self, key, value, encoder_len, y=None, mode='train'):

        B, key_seq_max_len, key_value_size = key.shape
        max_len = y.shape[1] if mode == 'train' else 600

        mask = torch.arange(key_seq_max_len).unsqueeze(0) >= torch.as_tensor(encoder_len).unsqueeze(1)
        mask = mask.to(self.args.device)
        
        predictions = torch.zeros((B, self.vocab_size, max_len)).to(self.args.device)
        prediction = torch.zeros(B, dtype=torch.long).to(device=self.args.device)

        hidden_states = [None, None] 
        
        context = torch.zeros(B, key_value_size).to(device=self.args.device)

        attention_plot = [] 
        attention_plot = torch.zeros(max_len, key_seq_max_len).to(self.args.device)

        for i in range(max_len):
            if mode == 'train':
                if y is not None and i > 0:
                    input = y[:, i-1]
                else:
                    input = prediction
                input = input.to(self.args.device)
                context, hidden_states, attention = self._forward_step(input, context, hidden_states, key, value, mask)

                attention_plot[i, :] = attention[0].detach().cpu()

            output_context = torch.cat([hidden_states[1], context], dim=1)
            
            prediction = self.character_prob(output_context)
            predictions[:, :, i] = prediction

        return predictions

class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, encoder_hidden_dim, decoder_hidden_dim, embed_dim, args, key_value_size=128):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, args, embed_dim, key_value_size)
        self.args = args

    def forward(self, x, x_len, y=None, mode='train'):
        key, value, encoder_len = self.encoder(x, x_len)
        #predictions, attentions = self.decoder(key, value, encoder_len, y=y, mode=mode)
        predictions = self.decoder(key, value, encoder_len, y=y, mode=mode)
        return predictions