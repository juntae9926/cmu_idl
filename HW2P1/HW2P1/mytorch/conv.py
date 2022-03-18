# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        # TODO
        B, C, I = A.shape
        self.batch_size = B
        self.input_size = I
        self.output_size = (self.input_size - self.kernel_size) + 1
        Z = np.zeros((self.batch_size, self.out_channels, self.output_size))

        for i in range(self.output_size):
            a = self.A[:,:,i:i+self.kernel_size]
            Z[:,:,i] = np.tensordot(a, self.W, axes=([1, 2], [1, 2])) + self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        
        #self.dLdb = np.sum(np.sum(dLdZ, axis=0), axis=1)
        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        self.dLdW = np.zeros((self.out_channels, self.in_channels, self.kernel_size))
        for i in range(self.kernel_size):
            a = self.A[:,:,i:i+self.output_size] 
            self.dLdW[:,:,i] = np.tensordot(dLdZ, a, axes=([0, 2],[0, 2]))
        
        dLdA = np.zeros((self.batch_size, self.in_channels, self.input_size))
        pad_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1)), constant_values=0)
        
        flip_W = np.flip(self.W, axis=2) # flipped_W
        for j in range(self.input_size):
            small_dLdZ = pad_dLdZ[:,:,j:j+self.kernel_size]
            dLdA[:,:,j] = np.tensordot(small_dLdZ, flip_W, axes=([1,2],[0,2]))
            
        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None) # TODO
        self.downsample1d = Downsample1d(downsampling_factor=stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        # TODO
        output = self.conv1d_stride1.forward(A)

        # downsample
        # TODO
        Z = self.downsample1d.forward(output)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        output = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        # TODO 
        dLdA = self.conv1d_stride1.backward(output)

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        #TODO
        self.batch_size, self.in_channels, self.width, self.height = A.shape
        self.output_size_W = (self.width - self.kernel_size) + 1
        self.output_size_H = (self.height - self.kernel_size) + 1
        Z = np.zeros((self.batch_size, self.out_channels, self.output_size_W, self.output_size_H))

        for i in range(self.output_size_H):
            for j in range(self.output_size_W):
                a = self.A[:,:,j:j+self.kernel_size,i:i+self.kernel_size]
                Z[:,:,j,i] = np.tensordot(a, self.W, axes=([1,2,3],[1,2,3])) + self.b



        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        self.dLdW = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                a = self.A[:,:,i:i+self.output_size_W,j:j+self.output_size_H]
                self.dLdW[:,:,i,j] = np.tensordot(dLdZ, a, axes=([0, 2, 3],[0, 2, 3]))
        
        dLdA = np.zeros((self.batch_size, self.in_channels, self.width, self.height))
        pad_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)), constant_values=0)

        flip_W = np.flip(self.W, axis=(2, 3))
        for i in range(self.width):
            for j in range(self.height):
                small_dLdZ = pad_dLdZ[:,:,i:i+self.kernel_size,j:j+self.kernel_size]
                dLdA[:,:,i,j] = np.tensordot(small_dLdZ, flip_W, axes=([1,2,3],[0,2,3]))

        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None) # TODO
        self.downsample2d = Downsample2d(downsampling_factor=stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        # TODO
        output = self.conv2d_stride1.forward(A)

        # downsample
        # TODO
        Z = self.downsample2d.forward(output)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        # TODO
        output = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        # TODO 
        dLdA = self.conv2d_stride1.backward(output)

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        #TODO
        self.upsample1d = Upsample1d(upsampling_factor) #TODO
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        #TODO
        # upsample
        A_upsampled = self.upsample1d.forward(A) #TODO
        
        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)  #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #TODO

        #Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ) #TODO

        dLdA =  self.upsample1d.backward(delta_out) #TODO

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) #TODO
        self.upsample2d = Upsample2d(upsampling_factor) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled =  self.upsample2d.forward(A)#TODO

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled) #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ) #TODO

        dLdA = self.upsample2d.backward(delta_out) #TODO

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """

        # TODO
        self.batch, self.channel, self.weight = A.shape
        Z = np.reshape(A, (self.batch, self.channel * self.weight))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        #TODO
        dLdA = np.reshape(dLdZ, (self.batch, self.channel, self.weight))

        return dLdA

