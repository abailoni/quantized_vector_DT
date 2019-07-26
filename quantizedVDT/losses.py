import numpy as np
import torch.nn as nn
import torch
from keras.utils import to_categorical
from inferno.extensions.criteria import SorensenDiceLoss

from speedrun.log_anywhere import log_scalar, log_image


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

    def __init__(self, n_channels, n_directions, weights=[1., 1.]):
        super().__init__()
        self.n_channels = n_channels
        self.n_directions = n_directions
        self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()
        self.weights = weights

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
        return self.weight[0]*loss1+self.weight[1]*loss2


class L1andSDloss(nn.Module):

    def __init__(self, n_channels, n_directions, weights=[1., 1.], log=True, exclude_borders=[0, 0, 0]):
        """

        :param n_channels:
        :param n_directions:
        :param weights: First gives weight for Sorensen Dice loss, second gives L1-Loss
        :param log:
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_directions = n_directions
        self.l1 = nn.L1Loss()
        self.sd = SorensenDiceLoss()
        self.weights = weights
        self.log = log
        self.exclude_borders = exclude_borders


    def forward(self, prediction, target):

        # exclude the spatial borders of the volume due to lacking context for the network
        # Not protected against 'bad' slicing
        b = self.exclude_borders
        prediction = prediction[..., b[0]:-(b[0]+1), b[1]:-(b[1]+1), b[2]:-(b[2]+1)]
        target = target[..., b[0]:-(b[0]+1), b[1]:-(b[1]+1), b[2]:-(b[2]+1)]


        one = prediction[:, :self.n_directions*self.n_channels].reshape(
            (1, self.n_directions, self.n_channels, *prediction.shape[2:]))
        one = one.permute(0, 2, 1, 3, 4, 5)
        # label = target[:, :self.n_directions].long()
        one_target = target[:, self.n_directions*self.n_channels:].reshape(
            (1, self.n_directions, self.n_channels, *prediction.shape[2:]))
        one_target = one_target.permute(0, 2, 1, 3, 4, 5)

        loss1 = self.sd(one, one_target)

        res_pred = prediction[:, self.n_directions * self.n_channels:]
        res_tar = target[:, self.n_directions:self.n_directions * self.n_channels]
        mask = target[:, self.n_directions * self.n_channels:-self.n_directions]

        loss2 = self.l1(res_pred*mask, res_tar*mask)

        if self.log:
            log_scalar('SorensenDiceLoss', self.weights[0]*loss1)
            log_scalar('L1Loss', self.weights[1]*loss2)

        return self.weights[0]*loss1+self.weights[1]*loss2











