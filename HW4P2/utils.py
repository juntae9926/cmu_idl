import torch
import Levenshtein
import numpy as np

from letter_list import LETTER_LIST

letter_list = LETTER_LIST

def calculate_levenshtein(x, y, len_x, len_y, decoder, PHONEME_MAP):

    beam_results, beam_scores, timesteps, out_lens = decoder.decode(x.permute(1, 0, 2), seq_lens=len_x)

    batch_size = beam_results.shape[0] # TODO

    dist = 0

    # for i in range(batch_size): # Loop through each element in the batch
    for i in range(batch_size):
        max_idx = torch.argmax(beam_scores[i])

        h_sliced = beam_results[i, max_idx, :out_lens[i, max_idx]] # [335]
        h_string = [PHONEME_MAP[i] for i in h_sliced]
        h_string = "".join(h_string)

        y_sliced = y[i, :len_y[i]]
        y_string = [str(PHONEME_MAP[i]) for i in y_sliced]
        y_string = "".join(y_string)

    dist/=batch_size

    return dist

def create_dictionaries(letter_list):
    letter2index = dict()
    index2letter = dict()

    for i,ii in enumerate(letter_list):
        index2letter[i]=ii
        letter2index[ii]=i
    return letter2index, index2letter

def levenshtein(x, y, len_x, len_y, batch_size):

    letter2index, index2letter = create_dictionaries(LETTER_LIST)

    dist = 0
    arg_max = torch.argmax(x, dim=1)
    batch_size = y.size(0)

    for i in range(batch_size):
        x_sliced = arg_max[i]
        len = (x_sliced == 29).nonzero(as_tuple=False)

        if len.size(0) != 0:
            x_sliced = x_sliced[:len[0]]
        else:
            x_sliced = x_sliced
        
        x_string = ''.join(index2letter[j.item()] for j in x_sliced[:])
        y_sliced =  y[i, :len_y[i]][1:-1]
        y_string =   ''.join(index2letter[j.item()] for j in y_sliced)
        dis = Levenshtein.distance(x_string, y_string)

        dist += dis
    dist /= batch_size

    return dist

