import torchvision
from inferno.extensions.models import UNet
from inferno.extensions.layers import ConvReLU2D
from inferno.io.volumetric import HDF5VolumeLoader
from inferno.io.core import Zip
from inferno.utils.io_utils import yaml2dict
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.nn.functional as F

import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from inferno.utils.python_utils import ensure_dir

LOG_DIRECTORY = ensure_dir('./logs_2')


BATCHSIZE = 8
N_DIRECTIONS = 8


# unsq = transforms.Lambda(lambda x: torch.unsqueeze(x, 0))
transpose = transforms.Lambda(lambda x: torch.transpose(x, 0, 1))
squeeze = transforms.Lambda(lambda x: torch.squeeze(x, 1))
trans = transforms.Compose([transforms.ToTensor(), transpose])
trans2 = transforms.Compose([transforms.ToTensor(), squeeze])

imageset_train = HDF5VolumeLoader(path='./train-volume.h5', path_in_h5_dataset='data',
                                  transforms=trans, **yaml2dict('config_train.yml')['slicing_config'])
labelset_train = HDF5VolumeLoader(path='./stardistance.h5', path_in_h5_dataset='data',
                                  transforms=trans2, **yaml2dict('config_train.yml')['slicing_config_truth'])
trainset = Zip(imageset_train, labelset_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCHSIZE,
                                          shuffle=True, num_workers=2)
imageset_val = HDF5VolumeLoader(path='./val-volume.h5', path_in_h5_dataset='data',
                                transforms=trans, **yaml2dict('config_val.yml')['slicing_config'])
labelset_val = HDF5VolumeLoader(path='./stardistance_val.h5', path_in_h5_dataset='data',
                                transforms=trans2, **yaml2dict('config_val.yml')['slicing_config_truth'])
trainset = Zip(imageset_val, labelset_val)
valloader = torch.utils.data.DataLoader(trainset, batch_size=BATCHSIZE,
                                        shuffle=True, num_workers=2)


net = torch.nn.Sequential(
    ConvReLU2D(in_channels=1, out_channels=3, kernel_size=3),
    UNet(in_channels=3, out_channels=N_DIRECTIONS, dim=2, final_activation='ReLU')
    )

trainer = Trainer(net)

trainer.bind_loader('train', trainloader)
trainer.bind_loader('validate', valloader)

trainer.save_to_directory('./checkpoints')
trainer.save_every((200, 'iterations'))
trainer.build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                log_images_every='never'), log_directory=LOG_DIRECTORY)



trainer.validate_every((200, 'iterations'), for_num_iterations=50)
#trainer.build_metric()


trainer.build_criterion(nn.L1Loss)
trainer.build_optimizer(optim.Adam, lr=1e-4, weight_decay=0.0005)

trainer.set_max_num_iterations(20000)


if torch.cuda.is_available():
    trainer.cuda()

print('starting training')
trainer.fit()



print('hi')

