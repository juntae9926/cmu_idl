import torch
import Levenshtein
from phonemes import PHONEME_MAP, PHONEMES
# this function calculates the Levenshtein distance 

def calculate_levenshtein(x, y, len_x, len_y, decoder, PHONEME_MAP):

    # h - ouput from the model. Probability distributions at each time step 
    # y - target output sequence - sequence of Long tensors
    # lh, ly - Lengths of output and target
    # decoder - decoder object which was initialized in the previous cell
    # PHONEME_MAP - maps output to a character to find the Levenshtein distance
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(x.permute(1, 0, 2), seq_lens=len_x)
    # [B, 100, T]

    # TODO: You may need to transpose or permute h based on how you passed it to the criterion
    # Print out the shapes often to debug

    # TODO: call the decoder's decode method and get beam_results and out_len (Read the docs about the decode method's outputs)
    # Input to the decode method will be h and its lengths lh 
    # You need to pass lh for the 'seq_lens' parameter. This is not explicitly mentioned in the git repo of ctcdecode.

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

        dist += Levenshtein.distance(h_string, y_string)

    #     h_sliced = # TODO: Get the output as a sequence of numbers from beam_results
    #     # Remember that h is padded to the max sequence length and lh contains lengths of individual sequences
    #     # Same goes for beam_results and out_lens
    #     # You do not require the padded portion of beam_results - you need to slice it with out_lens 
    #     # If it is confusing, print out the shapes of all the variables and try to understand

    #     h_string = # TODO: MAP the sequence of numbers to its corresponding characters with PHONEME_MAP and merge everything as a single string

    #     y_sliced = # TODO: Do the same for y - slice off the padding with ly
    #     y_string = # TODO: MAP the sequence of numbers to its corresponding characters with PHONEME_MAP and merge everything as a single string
        
    #     dist += Levenshtein.distance(h_string, y_string)

    dist/=batch_size

    return dist