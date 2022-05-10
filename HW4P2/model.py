import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence

class LockedDropout(nn.Module):
    def __init__(self, dim=1):
        super(LockedDropout,self).__init__()
        self.dim = dim

    def forward(self, x, dropout=0.5):
        if self.dim == 0:
            mask = torch.zeros((1, x.shape[1], x.shape[2]), requires_grad=False, device=x.device).bernoulli_(1-dropout)
        else:
            mask = torch.zeros((x.shape[0], 1, x.shape[2]), requires_grad=False, device=x.device).bernoulli_(1-dropout)

        return mask * x

class pBLSTM(nn.Module):
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
    def __init__(self, input_dim, encoder_hidden_dim, dropout=0.2, key_value_size=256):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, encoder_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.pBLSTMs = nn.Sequential(pBLSTM(encoder_hidden_dim * 2, encoder_hidden_dim * 2),
                                    pBLSTM(encoder_hidden_dim * 4, encoder_hidden_dim * 2),
                                    pBLSTM(encoder_hidden_dim * 4, encoder_hidden_dim * 2))
         
        self.key_network = nn.Linear(encoder_hidden_dim * 4, key_value_size)
        self.value_network = nn.Linear(encoder_hidden_dim * 4, key_value_size)

        self.dropout = LockedDropout(dropout)

    def forward(self, x, x_len):

        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x = self.pBLSTMs(x)
        x, lengths = pad_packed_sequence(x, batch_first=True)

        k = self.key_network(x)
        v = self.value_network(x)

        return k, v, lengths


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dim, dropout_p=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value, mask=None):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            mask = mask.reshape(-1, 1, score.shape[-1])
            score.masked_fill_(mask, -np.inf)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)
        context = torch.bmm(attn, value)

        return context, attn

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "attention dim should same as d_model"

        self.attn_dim = int(d_model / n_heads) # default:64
        self.n_heads = n_heads
        self.Linear_Q = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        self.Linear_K = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        self.Linear_V = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        self.scaled_dot_attn = ScaledDotProductAttention(self.attn_dim) # sqrt(d_k)

    def forward(self, q, k, v, mask=None):
        batch_size = v.size(0)

        # [Batch, Length, N, D] = [Batch, Length, 8, 64]
        query = self.Linear_Q(q).view(batch_size, -1, self.n_heads, self.attn_dim)
        key = self.Linear_K(k).view(batch_size, -1, self.n_heads, self.attn_dim)
        value = self.Linear_V(v).view(batch_size, -1, self.n_heads, self.attn_dim)

        # [Batch * N, Length, Dim]
        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)

        # mask
        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)

        context, attn = self.scaled_dot_attn(query, key, value, mask)
        context = context.view(self.n_heads, batch_size, -1, self.attn_dim)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.n_heads * self.attn_dim)

        return context, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, decoder_hidden_dim, args, embed_dim=512, att_dim=128, key_value_size=256):
        super(Decoder, self).__init__()
        self.hidden_dim = decoder_hidden_dim
        self.att_dim = att_dim
        # Hint: Be careful with the padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=29)
        self.activation = nn.ELU()

        self.attention = MultiHeadAttention(att_dim, n_heads=8)    

        # The number of cells is defined based on the paper
        self.lstm1 = nn.LSTMCell(input_size=(embed_dim+att_dim), hidden_size=decoder_hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=decoder_hidden_dim, hidden_size=key_value_size)

        self.vocab_size = vocab_size

        self.character_prob = nn.Linear(embed_dim, vocab_size) #: d_v -> vocab_size
        self.key_value_size = key_value_size
        
        self.character_prob.weight = self.embedding.weight # Weight tying
        self.args = args

    def _forward_step(self, input, h2, key, value, mask):

        embed = self.embedding(input)

        context, attn = self.attention(h2, key, value, mask)
        context = context.squeeze(1)
        attention_output = torch.cat([context, embed], dim=1)
        
        h1, _ = self.lstm1(attention_output)
        h2, _ = self.lstm2(h1)

        output = self.character_prob(torch.cat([h2, context], dim=1))
        output = self.activation(output)

        return output, h2

    def forward(self, key, value, encoder_len, y=None, mode='train'):

        B, key_seq_max_len, key_value_size = key.shape
        max_len = 600 if mode == 'test' else y.shape[1]

        mask = torch.arange(key_seq_max_len).unsqueeze(0) >= torch.as_tensor(encoder_len).unsqueeze(1) # B x T
        mask = mask.to(self.args.device)
        
        predictions = torch.zeros((B, self.vocab_size, max_len)).to(self.args.device)

        input = torch.zeros(B).long().to(self.args.device)
        output = torch.zeros(B, self.vocab_size).to(self.args.device)
        hidden = torch.rand(B, self.att_dim).to(self.args.device)

        for i in range(1, max_len):
            if mode == 'train':

                # teacher forcing
                prob = np.random.rand()
                if i > 1 and prob < 0.7:
                    input = y[:, i-1].long()
                else:
                    input = output.argmax(-1)
            else:
                input = output.argmax(-1)

            output, hidden = self._forward_step(input, hidden, key, value, mask)
            predictions[:, :, i] = output

        return predictions

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, vocab_size, encoder_hidden_dim, decoder_hidden_dim, embed_dim, args, key_value_size=256):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, args, embed_dim, key_value_size)
        self.args = args

    def forward(self, x, x_len, y=None, mode='train'):
        key, value, encoder_len = self.encoder(x, x_len)
        predictions = self.decoder(key, value, encoder_len, y=y, mode=mode)
        return predictions