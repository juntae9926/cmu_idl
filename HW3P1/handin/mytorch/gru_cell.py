import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h):
        return self.forward(x, h)

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx

        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh

        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx

        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        #print(self.Wzx.shape, self.x.shape, self.bzx.shape, self.Wzh.shape, self.hidden.shape, self.bzh.shape) # (20, 10) (10,), (20,) (20, 20) (20,) (20,)
        self.z = self.z_act(self.Wzx@self.x + self.bzx + self.Wzh@self.hidden + self.bzh) # (20,)
        self.r = self.r_act(self.Wrx@self.x + self.brx + self.Wrh@ self.hidden + self.brh) # (150,)
        self.n = self.h_act(self.Wnx@self.x + self.bnx + self.r*(self.Wnh@self.hidden + self.bnh)) # (140,)
        h_t = (1-self.z)*self.n + self.z*self.hidden

        
        # This code should not take more than 10 lines. 
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        # return h_t
        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        
        # This code should not take more than 25 lines.
        da_r = self.r_act.derivative() # shape: (2, ) [hidden,]
        da_z = self.z_act.derivative() # shape: (2, ) [hidden,]
        da_n = self.h_act.derivative(state = self.n) # shape: (2, ) [hidden,]

        # Compute all derivatives
        # STEP 1: using dLdh, compute dz dn dhid(little)
        dz = delta * (-self.n + self.hidden) # shape: (1, 2) [batch, hidden]
        dn = delta * (1 - self.z) # shape: (1, 2) [batch, hidden]
        dh = delta * self.z # shape: (1, 2) [batch, hidden]

        # STEP 2: using dn, compute dr dWnx dx dbnx dWnh dbnh dhid(little) 
        dr = (da_n*dn) * (self.Wnh@self.hidden + self.bnh) # shape: (1, 2) [batch, hidden]
        dx = (da_n*dn) @ self.Wnx  # shape: (1, 5) [batch, input]
        self.dWnx += (da_n*dn).T @ self.x.reshape(1, -1) # shape: (2, 5) [hidden, input]
        self.dbnx += (da_n*dn).reshape(-1) # (2, ) [hidden,]
        self.dWnh += ((da_n*dn) * self.r).T * self.hidden.reshape(1, -1) # (2, 2) [hidden, hidden]
        self.dbnh += ((da_n*dn) * self.r).reshape(-1) # (2,) [hidden,]
        dh += ((da_n*dn) * self.r) @ self.Wnh # [batch, hidden]

        # STEP 3: using dz, compute dx dWzx dbzx dWzh dbzh dhid(little)
        dx += (da_z*dz) @ self.Wzx # [batch, input]
        self.dWzx += (da_z*dz).T @ self.x.reshape(1, -1) # [hidden, input]
        self.dbzx += (da_z*dz).reshape(-1) # [hidden,]
        self.dWzh += (da_z*dz).T @ self.hidden.reshape(1, -1) # [hidden, hidden]
        self.dbzh += (da_z*dz).reshape(-1) # [hidden,]
        dh += (da_z*dz) @ self.Wzh # [batch, hidden]
        
        # STEP 4: using dr, compute dx dWrx dbrx dWrh dbrh dhid(final)
        dx += (da_r*dr) @ self.Wrx # [batch, input]
        self.dWrx += (da_r*dr).T @ self.x.reshape(1, -1) # [hidden, input]
        self.dbrx += (da_r*dr).reshape(-1) # [hidden,]
        self.dWrh += (da_r*dr).T @ self.hidden.reshape(1, -1) # [hidden, hidden]
        self.dbrh += (da_r*dr).reshape(-1) # [hidden,]
        dh += (da_r*dr) @ self.Wrh # [batch, hidden]

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        # return dx, dh
        return dx, dh