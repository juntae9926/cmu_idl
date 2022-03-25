import numpy as np
from activation import *


class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """
        RNN Cell forward (single time step).

        Input (see writeup for explanation)
        -----
        x: (batch_size, input_size)
            input at the current time step

        h: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns
        -------
        h_prime: (batch_size, hidden_size)
            hidden state at the current time step and current layer
        """

        """
        ht = tanh(Wihxt + bih + Whhht−1 + bhh) 
        """

        h_prime = self.activation(np.dot(x, self.W_ih.T) + self.b_ih + np.dot(h, self.W_hh.T) + self.b_hh) # TODO
         
        # return h_prime
        return h_prime

    def backward(self, delta, h, h_prev_l, h_prev_t):
        """
        RNN Cell backward (single time step).

        Input (see writeup for explanation)
        -----
        delta: (batch_size, hidden_size)
                Gradient w.r.t the current hidden layer

        h: (batch_size, hidden_size)
            Hidden state of the current time step and the current layer

        h_prev_l: (batch_size, input_size)
                    Hidden state at the current time step and previous layer

        h_prev_t: (batch_size, hidden_size)
                    Hidden state at previous time step and current layer

        Returns
        -------
        dx: (batch_size, input_size)
            Derivative w.r.t.  the current time step and previous layer

        dh: (batch_size, hidden_size)
            Derivative w.r.t.  the previous time step and current layer

        """
        # delta = (3, 20) (batch_size, hidden_size) (b, h)
        batch_size = delta.shape[0]
        
        # 0) Done! Step backward through the tanh activation function.
        # Note, because of BPTT, we had to externally save the tanh state, and
        # have modified the tanh activation function to accept an optionally input.
        dz = delta * self.activation.derivative(state = h) # TODO

        # 1) Compute the averaged gradients of the weights and biases
        self.dW_ih += np.matmul(dz.T, h_prev_l) / batch_size # TODO size: (b, h).T * (b, d) = (h, d)
        self.dW_hh += np.matmul(dz.T, h_prev_t) / batch_size # TODO size: (b, h).T * (b, h) = (h, h)
        self.db_ih += np.mean(dz, axis=0) # TODO
        self.db_hh += np.mean(dz, axis=0) # TODO

        # # 2) Compute dx, dh
        dx = np.matmul(dz, self.W_ih) # TODO size: (b, h) * (h, d) = (b, d)
        dh = np.matmul(dz, self.W_hh) # TODO size: (b, h) * (h, h) = (b, h)

        # 3) Return dx, dh
        # return dx, dh
        return dx, dh
