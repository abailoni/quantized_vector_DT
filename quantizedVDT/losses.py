import numpy as np
import torch.nn as nn
import torch
from keras.utils import to_categorical


class MultiLoss(nn.Module):

    def __init__(self, losslist, applyto_pred, applyto_target, weights=None):
        """

        :param losslist: List of losses
        :param applyto: How many chanels the losses are to be applied to
        """
        raise NotImplementedError
        super().__init__()
        self.losslist = losslist
        self.applyto_pred = [0]
        self.applyto_target = [0]
        self.applyto_pred += applyto_pred
        self.applyto_target += applyto_target
        self.weights = [1.]*len(self.losslist) if weights is None else weights

    def append(self, loss, applyto_pred, applyto_target, weight=None):
        self.losslist += [loss]
        self.applyto_pred += [applyto_pred]
        self.applyto_target += [applyto_target]
        self.weights += [1.] if weight is None else [weight]

    def forward(self, prediction, target):
        cumloss = 0

        for i in range(len(self.losslist)):
            start_p = self.applyto_pred[i]
            stop_p = self.applyto_pred[i]+self.applyto_pred[i+1]
            start_t = self.applyto_target[i]
            stop_t = self.applyto_target[i]+self.applyto_target[i+1]
            cumloss += self.losslist[i](prediction[:, start_p:stop_p], target[:, start_t:stop_t])*self.weights[i]
        return cumloss


class L1andCEloss(nn.Module):

    def __init__(self, n_channels, n_directions):
        super().__init__()
        self.n_channels = n_channels
        self.n_directions = n_directions
        self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        one = prediction[:, :self.n_directions*self.n_channels].reshape(
            (1, self.n_directions, self.n_channels, *prediction.shape[2:]))
        one = one.permute(0, 2, 1, 3, 4, 5)
        label = target[:, :self.n_directions].long()

        loss1 = self.ce(one, label)

        res_pred = prediction[:, self.n_directions*self.n_channels:]
        res_tar = target[:, self.n_directions:self.n_directions*self.n_channels]
        mask = target[:, self.n_directions*self.n_channels:-self.n_directions]

        # We need to create a mask so that the L1-Loss is only applied where the residual is needed,
        # meaning the residual to class n is only penaliized in places that belong to class n

        # mask = np.moveaxis(to_categorical(label, num_classes=self.n_channels), -1, 1
        #                       ).reshape((res_pred.shape[0], self.n_directions*self.n_channels, *res_pred.shape[2:])
        #                                 , order='C')[:, :-self.n_directions, ...]

        loss2 = self.l1(res_pred*mask, res_tar*mask)

        # print(loss1, loss2)
        return loss1+loss2



class MyL1(nn.L1Loss):

    def forward(self, input, target, mask):
        super(nn.L1Loss).forward(input*mask, target*mask)









