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
from stardist import star_dist
from neurofire.criteria.loss_wrapper import LossWrapper
from inferno.io.transform import Transform

LOG_DIRECTORY = ensure_dir('./logs_3')
BATCHSIZE = 8
N_DIRECTIONS = 8
OPENCL_AVAILABLE = True

class LabelToTarget(Transform):
    def __init__(self):
        super().__init__()

    def batch_function(self, tensors):
        prediction, target = tensors
        target = np.moveaxis(star_dist(target[0].numpy(), N_DIRECTIONS, opencl=OPENCL_AVAILABLE), -1, 0)
        return prediction, target

tosignedint = transforms.Lambda(lambda x: torch.tensor(np.int32(x), dtype=torch.int32))

sdist = transforms.Lambda(lambda x: np.moveaxis(star_dist(x[0], N_DIRECTIONS, opencl=OPENCL_AVAILABLE), -1, 0))
labeltotarget = transforms.Compose([sdist])

train_images = HDF5VolumeLoader(path='./train-volume.h5', path_in_h5_dataset='data',
                                **yaml2dict('config_train.yml')['slicing_config'])
train_labels = HDF5VolumeLoader(path='labeled_segmentation.h5', path_in_h5_dataset='data',
                                transforms=tosignedint, **yaml2dict('config_train.yml')['slicing_config'])
trainset = Zip(train_images, train_labels)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCHSIZE,
                                          shuffle=True, num_workers=2)


val_images = HDF5VolumeLoader(path='./val-volume.h5', path_in_h5_dataset='data',
                              **yaml2dict('config_val.yml')['slicing_config'])
val_labels = HDF5VolumeLoader(path='labeled_segmentation_validation.h5', path_in_h5_dataset='data',
                              transforms=tosignedint, **yaml2dict('config_val.yml')['slicing_config'])
valset = Zip(val_images, val_labels)
valloader = torch.utils.data.DataLoader(trainset, batch_size=BATCHSIZE,
                                        shuffle=True, num_workers=2)

criterion = LossWrapper(criterion=nn.L1Loss,
                        transforms=LabelToTarget())


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
# trainer.build_metric()


trainer.build_criterion(criterion)
trainer.build_optimizer(optim.Adam, lr=1e-4, weight_decay=0.0005)

trainer.set_max_num_iterations(20000)


if torch.cuda.is_available():
    trainer.cuda()

print('starting training')
trainer.fit()

print('finished training')

print('the end')
