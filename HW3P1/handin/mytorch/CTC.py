import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """
        
        Initialize instance variables

        Argument(s)
        -----------
        
        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK


    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------

        skip_connect = [False for _ in range(N)]
        for i in range(N):
            if (i > 2) & (extended_symbols[i] != 0) & (extended_symbols[i] != extended_symbols[i-2]):
                skip_connect[i] = True
                
        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        # return extended_symbols, skip_connect
        return extended_symbols, skip_connect


    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        #S, T = len(extended_symbols), len(logits)
        S = extended_symbols.shape[0]
        T = logits.shape[0]
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        # TODO: Intialize alpha[0][1]
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # <---------------------------------------------
        alpha[0][0] = logits[0][extended_symbols[0]]
        alpha[0][1] = logits[0][extended_symbols[1]]

        for t in range(1, T):
            alpha[t][0] = alpha[t-1][0] * logits[t][extended_symbols[0]] # prev alpha * curr logits
            for s in range(1, S):
                alpha[t][s] = alpha[t-1][s-1] + alpha[t-1][s]
                if skip_connect[s] == True:
                    alpha[t][s] += alpha[t-1][s-2]
                alpha[t][s] *= logits[t][extended_symbols[s]]

        # return alpha
        return alpha


    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
        
        """
        #S, T = len(extended_symbols), len(logits)
        S = extended_symbols.shape[0]
        T = logits.shape[0]
        beta = np.zeros(shape=(T, S)) # [12, 5]

        # -------------------------------------------->
        # TODO
        # <--------------------------------------------
        beta[T-1, S-1] = 1 # last beta [11, 4]
        beta[T-1, S-2] = 1 # [11, 3]
        for t in reversed(range(T-1)):
            beta[t, S-1] = beta[t+1, S-1] * logits[t+1, extended_symbols[S-1]]
            for s in reversed(range(S-1)):
                beta[t, s] = beta[t+1, s] * logits[t+1, extended_symbols[s]] + beta[t+1, s+1] * logits[t+1, extended_symbols[s+1]]
                if (s < S-3) and (skip_connect[s+2]):
                    beta[t, s] += beta[t+1, s+2] * logits[t+1, extended_symbols[s+2]]

        # return beta
        return beta
		

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------

        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=1).reshape((-1, 1))

        # return gamma
        return gamma


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        
        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)


    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        # seq_len: 15, batch_size:12, 
        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------
            ib = logits[:input_lengths[b], b, :] # truncated_logits with input_lengths
            tb = target[b, :target_lengths[b]] # truncated_target with target_lengths
            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(tb) 
            alpha = self.ctc.get_forward_probs(ib, extended_symbols, skip_connect) 
            beta = self.ctc.get_backward_probs(ib, extended_symbols, skip_connect)
            gamma = self.ctc.get_posterior_probs(alpha, beta) # (15, 9) [input_len, 2 * target_len + 1]
            self.extended_symbols.append(extended_symbols)
            self.gammas.append(gamma)

            for s in range(len(extended_symbols)):
                # Caution: index of logits' symbol should be used extended_symbol index
                total_loss[b] -= np.sum(gamma[:, s] * np.log(ib[:, extended_symbols[s]])) 

        total_loss = np.sum(total_loss) / B
        
        # return total_loss
        return total_loss
		

    def backward(self):
        """
        
        CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            ib = self.logits[:self.input_lengths[b], b, :]
            tb = self.target[b, :self.target_lengths[b]]
            extended_symbols, _ = self.ctc.extend_target_with_blank(tb) 
            for s in range(len(extended_symbols)):
                dY[:self.input_lengths[b], b, extended_symbols[s]] -= self.gammas[b][:, s] / ib[:, extended_symbols[s]]

        # return dY
        return dY
