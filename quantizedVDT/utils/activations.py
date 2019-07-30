import numpy as np
import torch 
import torch.nn as nn


class PartialSoftmax(nn.Softmax):

    def __init__(self, dim=None, nclasses=4, ndirs=8):

        if dim is None:
            dim = -4
        super(PartialSoftmax, self).__init__(dim=dim)

        self.nclasses = nclasses
        self.ndirs = ndirs


    def forward(self, input):
        shape = input.shape

        classes = input[:, :8*4].reshape(shape[0], self.ndirs, self.nclasses, *shape[2:])

        super(PartialSoftmax, self).forward(classes)

        input[:, :self.ndirs*self.nclasses] = classes.reshape(shape[0], self.ndirs*self.nclasses, *shape[2:])

        return input

