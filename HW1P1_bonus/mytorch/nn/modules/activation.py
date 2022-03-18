import numpy as np


class Identity:
    
    def forward(self, Z):
    
        self.A = Z
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.ones(self.A.shape, dtype="f")
        
        return dAdZ


class Sigmoid:
    
    def forward(self, Z):
    
        self.A = 1 / (1 + np.exp(-Z)) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = self.A - np.square(self.A) # TODO
        
        return dAdZ


class Tanh:
    
    def forward(self, Z):
    
        self.A = np.tanh(Z) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = 1 - np.square(self.A) # TODO
        
        return dAdZ


class ReLU:
    
    def forward(self, Z):
    
        self.A = np.where(Z < 0, 0.0, Z) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.where(self.A > 0, 1, self.A) # TODO
        
        return dAdZ
        
        
