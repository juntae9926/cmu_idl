import torch
import Levenshtein
import numpy as np

from letter_list import LETTER_LIST

letter_list = LETTER_LIST

def create_dictionaries(letter_list):
    '''
    Create dictionaries for letter2index and index2letter transformations
    based on LETTER_LIST

    Args:
        letter_list: LETTER_LIST

    Return:
        letter2index: Dictionary mapping from letters to indices
        index2letter: Dictionary mapping from indices to letters
    '''
    letter2index = dict()
    index2letter = dict()
    # TODO
    return letter2index, index2letter
    

def transform_index_to_letter(batch_indices):
    '''
    Transforms numerical index input to string output by converting each index 
    to its corresponding letter from LETTER_LIST

    Args:
        batch_indices: List of indices from LETTER_LIST with the shape of (N, )
    
    Return:
        transcripts: List of converted string transcripts. This would be a list with a length of N
    '''
    transcripts = []
    # TODO
    for i in batch_indices:
        transcripts.append(letter_list[i])
    return transcripts
        
# Create the letter2index and index2letter dictionary
letter2index, index2letter = create_dictionaries(LETTER_LIST)

def calculate_levenshtein(x, y, len_x, len_y, decoder, PHONEME_MAP):

    # h - ouput from the model. Probability distributions at each time step 
    # y - target output sequence - sequence of Long tensors
    # lh, ly - Lengths of outpu t and target
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

def levenshtein(seq1, seq2):  
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])