# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class BatchNorm2d:

    def __init__(self, num_features, alpha=0.9):
        # num features: number of channels
        self.alpha = alpha
        self.eps = 1e-8

        self.Z = None
        self.NZ = None
        self.BZ = None

        self.BW = np.ones((1, num_features, 1, 1))
        self.Bb = np.zeros((1, num_features, 1, 1))
        self.dLdBW = np.zeros((1, num_features, 1, 1))
        self.dLdBb = np.zeros((1, num_features, 1, 1))

        self.M = np.zeros((1, num_features, 1, 1))
        self.V = np.ones((1, num_features, 1, 1))

        # inference parameters
        self.running_M = np.zeros((1, num_features, 1, 1))
        self.running_V = np.ones((1, num_features, 1, 1))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        if eval:
            self.Z = Z
            NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            self.BZ = self.BW * NZ + self.Bb
            return self.BZ
        
        B, C, W, H = Z.shape

        self.Z = Z
        self.N = B  # the numebr of batch
        # self.M = 1/self.N * np.sum(Z, axis=(0, 1)) # TODO
        # self.V = np.sum(np.square(self.Z - self.M), axis=(0, 1))  # TODO
        Z_temp = self.Z.transpose(1, 0, 2, 3) # [C, B, W, H]
        for c in range(C):
            M_temp = np.sum(Z_temp[c]) / (B*W*H)
            self.M[0, c, 0, 0] = M_temp

        for c in range(C):
            V_temp = np.sum(np.square(self.Z[:, c, :, :] - self.M[0, c, 0, 0])) / (B*W*H)
            self.V[0, c, 0, 0] = V_temp

        self.NZ = (self.Z - self.M) / np.sqrt(self.V + self.eps)  # TODO
        self.BZ = self.BW * self.NZ + self.Bb  # TODO

        self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M    # TODO
        self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V  # TODO

        return self.BZ

    def backward(self, dLdBZ):
        # self.dLdBW = np.sum(dLdBZ * self.NZ, axis=(0, 2, 3), keepdims=True)  # TODO
        # self.dLdBb = np.sum(dLdBZ, axis=(0, 2, 3), keepdims=True)  # TODO

        # dLdNZ = dLdBZ * self.BW  # TODO
        # dLdV = -1/2 * np.sum((dLdNZ * (self.Z - self.M) * ((self.V + self.eps)**(-3/2))), axis=(0, 2, 3), keepdims=True) # TODO

        # dLdM = - np.sum(dLdNZ * (self.V + self.eps)**(-1/2), axis=(0, 2, 3), keepdims=True) \
        #        + np.sum(dLdNZ * (self.Z - self.M) * (self.V + self.eps) ** (-1.5), axis=(0, 2, 3), keepdims=True) \
        #        * np.sum((self.Z - self.M), axis=(0, 2, 3), keepdims=True)  # TODO

        # dLdZ = dLdNZ * (self.V + self.eps)**(-1/2) + dLdV * (2/self.N * (self.Z - self.M)) + dLdM * (1/(self.N * self.Z.shape[2] * self.Z.shape[3]))   # TODO

        self.dLdBW  = np.sum(dLdBZ * self.NZ, axis=(0, 2, 3), keepdims=True) # TODO
        self.dLdBb  = np.sum(dLdBZ, axis=(0, 2, 3), keepdims=True) # TODO
        
        dLdNZ       = dLdBZ * self.BW # TODO
        dLdV        = -1/2 * np.sum((dLdNZ * (self.Z - self.M) * ((self.V + self.eps)**(-3/2))), axis=(0, 2, 3), keepdims=True)  # TODO
        dLdM        = - np.sum(dLdNZ * (self.V + self.eps)**(-1/2), axis=(0, 2, 3), keepdims=True) -2/(self.N*self.Z.shape[2]*self.Z.shape[3]) * dLdV * np.sum((self.Z - self.M), axis=(0, 2, 3), keepdims=True)  # TODO
        dLdZ        = dLdNZ * (self.V + self.eps)**(-1/2) + dLdV * (2/(self.N*self.Z.shape[2]*self.Z.shape[3]) * (self.Z - self.M)) + dLdM * (1/(self.N*self.Z.shape[2]*self.Z.shape[3])) # TODO

        return dLdZ
