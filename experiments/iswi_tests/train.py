import torchvision
from inferno.extensions.models import UNet
from inferno.io.volumetric import HDF5VolumeLoader
from inferno.io.core import Zip

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms

from speedrun import BaseExperiment, TensorboardMixin

imageloader = HDF5VolumeLoader(path='./train-volume.h5', path_in_h5_dataset='data', data_slice='0:10, 0:100, 0:100')
labelloader = HDF5VolumeLoader(path='/dist_trans.h5', path_in_h5_dataset='data', data_slice='0:10, 0:100, 0:100')
trainloader = Zip(imageloader, labelloader)

class MyExperiment(BaseExperiment, TensorboardMixin):
    def __init__(self):
        super(MyExperiment, self).__init__()
        self.auto_setup()
        self.net = UNet(in_channels=1, out_channels=1, dim=3)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.trainloader = trainloader
        self.optimizer = optim.Adam(self.net.parameters())
        self.criterion = nn.CrossEntropyLoss()


    def run(self):
        running_loss = 0.0
        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (1, i + 1, running_loss / 2000))
                running_loss = 0.0


if __name__ == '__main__':
    experiment = MyExperiment()
    experiment.run()
