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

    def __init__(self, n_channels, n_directions, weights=[1., 1.], log=True, exclude_borders=[0, 0, 0], max_dist=100):
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
        self.max_dist = max_dist
        self.log_counter = 0


    def forward(self, prediction, target):
        # shape of prediction: [batch, n_dir*n_channels*2, z, y, x]; e.g. [1, 64, 12, 324, 324]

        if self.log_counter % 100 == 0:
            log_now = True
        else:
            log_now = False
        self.log_counter += 1

        # exclude the spatial borders of the volume due to lacking context for the network
        # Not protected against 'bad' slicing
        if type(self.exclude_borders) is list:
            b = self.exclude_borders
            prediction = prediction[..., b[0]:-(b[0]+1), b[1]:-(b[1]+1), b[2]:-(b[2]+1)]
            target = target[..., b[0]:-(b[0]+1), b[1]:-(b[1]+1), b[2]:-(b[2]+1)]
        if log_now:
            log_image('target_unmasked', target)

        if self.exclude_borders == 'auto':
            mask_res = np.ones((prediction.shape[0], (self.n_channels-1)*self.n_directions, *prediction.shape[2:]))
            mask_class = np.ones((prediction.shape[0], self.n_directions*self.n_channels, *prediction.shape[2:]))
            for i in range(self.n_directions):
                xoffset = -int(np.cos(i/self.n_directions*2*np.pi)*self.max_dist)
                yoffset = -int(np.sin(i/self.n_directions*2*np.pi)*self.max_dist)
                xslice = slice(xoffset-1, None) if xoffset < 0 else slice(None, xoffset+1)
                yslice = slice(yoffset-1, None) if yoffset < 0 else slice(None, yoffset+1)
                mask_class[:, self.n_channels*i:self.n_channels*(i+1), :, yslice, :] = 0
                # This is wrong
                # mask_res[:, (self.n_channels-1)*i:
                #          (self.n_channels-1)*(i+1), :, yslice, :] = 0
                # mask_res[:, (self.n_channels - 1) * i:
                #          (self.n_channels-1)*(i+1), :, :, xslice] = 0
                mask_res[:, i::self.n_directions, :, yslice, :] = 0
                mask_res[:, i::self.n_directions, :, :, xslice] = 0


                mask_class[:, self.n_channels*i:self.n_channels*(i+1), :, :, xslice] = 0
            mask_class = torch.Tensor(mask_class).cuda()
            mask_res = torch.Tensor(mask_res).cuda()
            prediction[:, :self.n_channels*self.n_directions] *= mask_class
            prediction[:, self.n_channels*self.n_directions:] *= mask_res

            target[:, self.n_directions*self.n_channels:] *= mask_class
            target[:, self.n_directions:self.n_directions*self.n_channels] *= mask_res
            # target[:, :-self.n_directions] = target[:, :-self.n_directions]*mask
        if log_now:
            log_image('mask_class', mask_class)
            log_image('mask_res', mask_res)
            log_image('pred', prediction)
            log_image('target_masked', target)

        one = prediction[:, :self.n_directions*self.n_channels].reshape(
            (1, self.n_directions, self.n_channels, *prediction.shape[2:]))
        one = one.permute(0, 2, 1, 3, 4, 5)
        # label = target[:, :self.n_directions].long()
        one_target = target[:, self.n_directions*self.n_channels:].reshape(
            (1, self.n_directions, self.n_channels, *prediction.shape[2:]))
        one_target = one_target.permute(0, 2, 1, 3, 4, 5)


        one[:, self.n_channels-1] *= -1
        one[:, self.n_channels-1] += 1
        one_target[:, self.n_channels-1] *= -1
        one_target[:, self.n_channels-1] += 1
        #
        # one_new = one*1.
        # one_new[:, self.n_channels-1] = 1-one[:, self.n_channels-1]


        if log_now:
            log_image('one', one)
            log_image('one_target', one_target)
            log_image('one_new', one)


        loss1 = self.sd(one, one_target)

        one[:, self.n_channels-1] *= -1
        one[:, self.n_channels-1] += 1


        res_pred = prediction[:, self.n_directions * self.n_channels:]
        res_tar = target[:, self.n_directions:self.n_directions * self.n_channels]

        # wrong
        # mask_quant = target[:, self.n_directions * self.n_channels:-self.n_directions]

        # mask_quant_int = target[:, self.n_directions * self.n_channels:]


        mask_quant = torch.empty(res_tar.shape, device='cuda')
        for i in range(self.n_directions):
            mask_quant[:, i::self.n_directions] = target[:, self.n_directions * self.n_channels + i*(self.n_channels):
                                                         self.n_directions * self.n_channels + (i+1)*(self.n_channels)-1]



        loss2 = self.l1(res_pred*mask_quant, res_tar*mask_quant)
        #
        if log_now:
            log_image('res_pred', res_pred)
            log_image('res_tar', res_tar)
            log_image('mask_quant', mask_quant)
            log_image('res_pred_masked', res_pred*mask_quant)
            log_image('res_tar_masked', res_tar*mask_quant)


        if self.log:
            log_scalar('SorensenDiceLoss', self.weights[0]*loss1)
            log_scalar('L1Loss', self.weights[1]*loss2)
            # log_image('test', mask_res[0, :, :, :, :])

        return self.weights[0]*loss1+self.weights[1]*loss2



class MaskedL1Loss(nn.L1Loss):

    def __init__(self, n_dir, a_max):
        super().__init__()
        self.n_dir = n_dir
        self.a_max = a_max
        self.mask = None

    def forward(self, input, target):

        if self.mask is None:
            mask = np.ones(input.shape)
            for i in range(self.n_dir):
                xoffset = -int(np.cos(i / self.n_dir * 2 * np.pi) * self.a_max)
                yoffset = -int(np.sin(i / self.n_dir * 2 * np.pi) * self.a_max)
                xslice = slice(xoffset - 1, None) if xoffset < 0 else slice(None, xoffset + 1)
                yslice = slice(yoffset - 1, None) if yoffset < 0 else slice(None, yoffset + 1)
                mask[:, i, :, yslice, :] = 0
                mask[:, i, :, :, xslice] = 0
            self.mask = torch.Tensor(mask).cuda()
        log_image('inp', input)
        log_image('inp_masked', input*self.mask)
        log_image('tar', target)
        log_image('tar_masked', target*self.mask)

        return super().forward(input*self.mask, target*self.mask)





