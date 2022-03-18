import numpy as np
from torch import strided
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height) (1, 9, 86, 86)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        self.batch_size, self.in_channels, self.width, self.height = A.shape # (1, 9, 86, 86)
        self.output_W = (self.width - self.kernel) + 1 # 81
        self.output_H = (self.height - self.kernel) + 1 # 81
        self.index = np.zeros((self.batch_size, self.in_channels, self.output_W, self.output_H))

        Z = np.zeros((self.batch_size, self.in_channels, self.output_W, self.output_H)) # (1, 9, 81, 81)
        for i in range(self.output_W):
            for j in range(self.output_H):
                Z[:,:,i,j] = np.max(A[:,:,i:i+self.kernel, j:j+self.kernel], axis=(2,3))
                for batch_i in range(self.batch_size):
                    for channel_i in range(self.in_channels):
                        idx = np.argmax(A[batch_i,channel_i,i:i+self.kernel, j:j+self.kernel])
                        self.index[batch_i, channel_i, i,j] = idx

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros((self.batch_size, self.in_channels, self.width, self.height)) # (1, 9, 86, 86)
        for i in range(self.output_W):
            for j in range(self.output_H):
                for batch_i in range(self.batch_size):
                    for channel_i in range(self.in_channels):
                        idx = self.index[batch_i, channel_i, i, j]
                        idx = int(idx.item())
                        x, y = np.unravel_index(idx, (self.kernel, self.kernel))
                        dLdA[batch_i,channel_i,i+x,j+y] += dLdZ[batch_i,channel_i, i, j]

        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.batch_size, self.in_channels, self.width, self.height = A.shape
        self.output_W = (self.width - self.kernel) + 1 # 81
        self.output_H = (self.height - self.kernel) + 1 # 81
        self.index = np.zeros((self.batch_size, self.in_channels, self.output_W, self.output_H))

        Z = np.zeros((self.batch_size, self.in_channels, self.output_W, self.output_H)) # (1, 9, 81, 81)
        for i in range(self.output_W):
            for j in range(self.output_H):
                Z[:,:,i,j] = np.mean(A[:,:,i:i+self.kernel, j:j+self.kernel], axis=(2, 3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros((self.batch_size, self.in_channels, self.width, self.height))
        for i in range(self.output_W):
            for j in range(self.output_H):
                for batch_i in range(self.batch_size):
                    for channel_i in range(self.in_channels):
                        mean = dLdZ[batch_i, channel_i, i, j] / (self.kernel * self.kernel)
                        dLdA[batch_i, channel_i, i:i+self.kernel, j:j+self.kernel] += np.full((self.kernel, self.kernel), mean)
        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel) #TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        output = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(output)
        
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        output = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(output)
        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel) #TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        output = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(output)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        output = self.downsample2d.backward(dLdZ)
        dLdZ = self.meanpool2d_stride1.backward(output)
        return dLdZ
