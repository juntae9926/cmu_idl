from os import TMP_MAX
import numpy as np
import pdb

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # TODO
        B, C, W = A.shape
        sampled_size = W*self.upsampling_factor - (self.upsampling_factor - 1)
        tmp = np.zeros((B, C, sampled_size))
        tmp[:,:,::self.upsampling_factor] = A[:,:,:]
        Z = tmp

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        #TODO
        B, C, W = dLdZ.shape
        recover_size = int((W + self.upsampling_factor - 1)/self.upsampling_factor)
        tmp = np.zeros((B, C, recover_size))
        tmp = dLdZ[:,:,::self.upsampling_factor]
        dLdA = tmp

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # TODO
        B, C, W = A.shape
        self.input_width = W
        downsample_size = int((W + self.downsampling_factor - 1)/self.downsampling_factor)
        tmp = np.zeros((B, C, downsample_size))
        tmp = A[:,:,::self.downsampling_factor]
        Z = tmp

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        #TODO
        B, C, W = dLdZ.shape
        recover_size = self.input_width
        tmp = np.zeros((B, C, recover_size))
        tmp[:,:,::self.downsampling_factor] = dLdZ[:,:,:]
        dLdA = tmp

        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        # TODO
        B, C, W, H = A.shape
        sampling_size_W = W*self.upsampling_factor - (self.upsampling_factor - 1)
        sampling_size_H = H*self.upsampling_factor - (self.upsampling_factor - 1)
        tmp = np.zeros((B,C,sampling_size_W, sampling_size_H))
        tmp[:,:,::self.upsampling_factor,::self.upsampling_factor] = A[:,:,:,:]
        Z = tmp

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        #TODO
        B, C, W, H = dLdZ.shape
        recover_size_W = int((W + self.upsampling_factor - 1)/self.upsampling_factor)
        recover_size_H = int((H + self.upsampling_factor - 1)/self.upsampling_factor)
        tmp = np.zeros((B,C,recover_size_W, recover_size_H))
        tmp[:,:,:,:] = dLdZ[:,:,::self.upsampling_factor,::self.upsampling_factor]
        dLdA = tmp

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        # TODO
        B, C, W, H = A.shape
        self.input_width = W
        self.input_height = H
        downsample_size_W = int((W + self.downsampling_factor - 1)/self.downsampling_factor)
        downsample_size_H = int((H + self.downsampling_factor - 1)/self.downsampling_factor)
        tmp = np.zeros((B, C, downsample_size_W, downsample_size_H))
        tmp = A[:,:,::self.downsampling_factor,::self.downsampling_factor]
        Z = tmp

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        #TODO

        B, C, W, H = dLdZ.shape
        recover_size_W = self.input_width
        recover_size_H = self.input_height
        tmp = np.zeros((B, C, recover_size_W, recover_size_H))
        tmp[:,:,::self.downsampling_factor,::self.downsampling_factor] = dLdZ[:,:,:,:]
        dLdA = tmp

        return dLdA