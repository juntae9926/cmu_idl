# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Dropout2d(object):
    def __init__(self, p=0.5):
        # Dropout probability
        self.p = p

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, eval=False):
        """
        Arguments:
          x (np.array): (batch_size, in_channel, input_width, input_height)
          eval (boolean): whether the model is in evaluation mode
        Return:
          np.array of same shape as input x
        """
        # 1) Get and apply a per-channel mask generated from np.random.binomial
        # 2) Scale your output accordingly
        # 3) During test time, you should not apply any mask or scaling.
        B, C, W, H = x.shape
        if eval == False:
          mask = []
          for _ in range(B):
            temp = []
            sample = np.random.binomial(1, self.p, C)
            for i in sample:
              temp_c = np.ones((W, H)) if i == 0 else np.zeros((W, H))
              temp.append(temp_c)
            temp = np.reshape(temp, (C, W, H))
            mask.append(temp)

          self.mask = np.reshape(mask, (x.shape))
          x = (x * mask) / (1 - self.p)

        return x

    def backward(self, delta):
        """
        Arguments:
          delta (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
          np.array of same shape as input delta
        """
        # 1) This method is only called during training.
        # 2) You should scale the result by chain rule

        return (delta * self.mask) / (1 - self.p)

