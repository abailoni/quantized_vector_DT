import torchvision
from inferno.extensions.models import UNet
from inferno.io.volumetric import HDF5VolumeLoader
from inferno.io.core import Zip
from inferno.utils.io_utils import yaml2dict
from inferno.trainers.basic import Trainer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.nn.functional as F

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

transpose = transforms.Lambda(lambda x: torch.transpose(x, 0, 1))
squeeze = transforms.Lambda(lambda x: torch.squeeze(x, 1))
fromnumpy = transforms.Lambda(lambda x: torch.from_numpy(x))
trans = transforms.Compose([fromnumpy])
trans2 = transforms.Compose([fromnumpy, squeeze])

imageset_val = HDF5VolumeLoader(path='./val-volume.h5', path_in_h5_dataset='data',
                                transforms=trans, **yaml2dict('config_val.yml')['slicing_config'])
labelset_val = HDF5VolumeLoader(path='./stardistance_val.h5', path_in_h5_dataset='data',
                                transforms=trans2, **yaml2dict('config_val.yml')['slicing_config_truth'])


trainer = Trainer()

trainer.load(from_directory='checkpoints', map_location='cpu', best=False)

result, loss = trainer.apply_model_and_loss(
    imageset_val[5].unsqueeze(0).unsqueeze(0),#.to('cuda'),
    labelset_val[5].unsqueeze(0).unsqueeze(0))#.to('cuda'))

print(loss)

print(result)


plt.figure()
plt.imshow(result.squeeze().detach().numpy()[0])
#plt.imshow(result[0].detach().squeeze().cpu().numpy())
plt.title('result')
plt.figure()
plt.imshow(imageset_val[5].detach().squeeze().cpu().numpy())
plt.title('image')
plt.figure()
plt.imshow(labelset_val[5].detach().squeeze().cpu().numpy()[0])
plt.title('ground truth')

plt.show()
