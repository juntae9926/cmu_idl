import numpy as np


def clean_path(path):
	""" utility function that performs basic text cleaning on path """

	# No need to modify
	path = str(path).replace("'","")
	path = path.replace(",","")
	path = path.replace(" ","")
	path = path.replace("[","")
	path = path.replace("]","")

	return path


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1
        
        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # print(y_probs.shape) # (4, 10, 1) [len(symbols) + 1, seq_length, batch_size]
        
        merged_length, seq_length, batch_size = y_probs.shape
        for b in range(batch_size):
            for s in range(seq_length):
                path_prob *= np.max(y_probs[:,s,b])
                idx = np.argmax(y_probs[:,s,b])
                if idx != 0:
                    if blank:
                        n = self.symbol_set[idx-1]
                        decoded_path.append(n)
                        blank = 0
                    else:
                        if s == 0 or (decoded_path[-1] != self.symbol_set[idx-1]):
                            decoded_path.append(self.symbol_set[idx-1])
                else: 
                    blank = 1

        decoded_path = clean_path(decoded_path)

        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def initialize(self, y_probs):

        blank_path = [""]
        blank_score = {"":y_probs[0,0,0]}
        symbol_path = []
        symbol_score = {}

        for i, s in enumerate(self.symbol_set):
            symbol_path.append(s)
            symbol_score[s] = y_probs[i + 1, 0, 0]
        
        return blank_path, blank_score, symbol_path, symbol_score

    def pruning(self, blank_path, blank_score, symbol_path, symbol_score, beam_width):
        score_list = []
        for value in blank_score.values():
            score_list.append(value)

        for value in symbol_score.values():
            score_list.append(value)

        score_list.sort()

        if len(score_list) < beam_width:
            cutoff = score_list[-1]
        else:
            cutoff = score_list[-beam_width]

        pruned_blank_path = []
        pruned_blank_score = {}
        pruned_symbol_path = []
        pruned_symbol_score = {}

        for p in blank_path:
            if blank_score[p] >= cutoff:
                pruned_blank_path.append(p)
                pruned_blank_score[p] = blank_score[p]

        for p in symbol_path:
            if symbol_score[p] >= cutoff:
                pruned_symbol_path.append(p)
                pruned_symbol_score[p] = symbol_score[p]
        
        return pruned_blank_path, pruned_blank_score, pruned_symbol_path, pruned_symbol_score


    def extend_with_blank(self, y_probs, blank_path, blank_score, symbol_path, symbol_score, s):
        
        extended_blank_path = []
        extended_blank_score = {}

        for p in blank_path:
            extended_blank_path.append(p)
            extended_blank_score[p] = blank_score[p] * y_probs[0, s, 0]
        
        for p in symbol_path:
            if p in extended_blank_path:
                extended_blank_score[p] += symbol_score[p] * y_probs[0, s, 0]
            else:
                extended_blank_path.append(p)
                extended_blank_score[p] = symbol_score[p] * y_probs[0, s, 0]
        
        return extended_blank_path, extended_blank_score


    
    def extend_with_symbol(self, y_probs, blank_path, blank_score, symbol_path, symbol_score, s):
        
        extended_symbol_path = []
        extended_symbol_score = {}
        for p in blank_path:
            for i, c in enumerate(self.symbol_set):
                new_path = p + c
                extended_symbol_path.append(new_path)
                extended_symbol_score[new_path] = blank_score[p] * y_probs[i+1, s, 0]
                
        for p in symbol_path:
            for i, c in enumerate(self.symbol_set):
                new_path = p if c == p[-1] else p + c
                if new_path in extended_symbol_path:
                    extended_symbol_score[new_path] += symbol_score[p] * y_probs[i+1, s, 0]
                else:
                    extended_symbol_path.append(new_path)
                    extended_symbol_score[new_path] = symbol_score[p] * y_probs[i+1, s, 0]
        
        return extended_symbol_path, extended_symbol_score
    
    def merge(self, blank_path, blank_score, symbol_path, symbol_score):

        paths = blank_path
        scores = blank_score
        
        for p in symbol_path:
            if p in paths:
                scores[p] += symbol_score[p]
            else:
                paths.append(p)
                scores[p] = symbol_score[p]

        best_scores = dict(sorted(scores.items(), key=lambda x: x[1]))
        best_path = list(best_scores.keys())[-1]

        return best_path, best_scores


    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        best_path, merged_path_scores = None, None
        seq_length = y_probs.shape[1]

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        #    - initialize a list to store all candidates
        # 2. Iterate over 'sequences'
        # 3. Iterate over symbol probabilities
        #    - Update all candidates by appropriately compressing sequences
        #    - Handle cases when current sequence is empty vs. when not empty
        # 4. Sort all candidates based on score (descending), and rewrite 'ordered'
        # 5. Update 'sequences' with first self.beam_width candidates from 'ordered'
        # 6. Merge paths in 'ordered', and get merged paths scores
        # 7. Select best path based on merged path scores, and return     

        #initialization
        blank_path, blank_score, symbol_path, symbol_score = self.initialize(y_probs)

        for s in range(1, seq_length):

            #pruning
            blank_path, blank_score, symbol_path, symbol_score = self.pruning(blank_path, blank_score, symbol_path, symbol_score, self.beam_width)
        
            #extend_with_blank
            extended_blank_path, extended_blank_score = self.extend_with_blank(y_probs, blank_path, blank_score, symbol_path, symbol_score, s)

            #extend_with_symbol
            extended_symbol_path, extended_symbol_score = self.extend_with_symbol(y_probs, blank_path, blank_score, symbol_path, symbol_score, s)

            #update
            blank_path, blank_score, symbol_path, symbol_score = extended_blank_path, extended_blank_score, extended_symbol_path, extended_symbol_score

        #merge
        best_path, merged_path_scores = self.merge(blank_path, blank_score, symbol_path, symbol_score)

        return best_path, merged_path_scores
