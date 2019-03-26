import torchvision
from inferno.extensions.models import UNet
from inferno.extensions.layers import ConvReLU2D, ConvReLU3D
from inferno.io.volumetric import HDF5VolumeLoader
from inferno.io.core import Zip
from inferno.utils.io_utils import yaml2dict
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.extensions.layers.sampling import AnisotropicUpsample

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

LOG_DIRECTORY = ensure_dir('./logs_3')
BATCHSIZE = 2
N_DIRECTIONS = 8
OPENCL_AVAILABLE = True

def sdist_volume(vol, n_directions):
    """
    returns the n-distances
    :param vol: np-like 3d (z,y,x)
            n_directions: number of directions
    :return: (8, z, y, x)
    """
    directions = np.empty((n_directions, *vol.shape), dtype=np.float32)
    for z in range(vol.shape[0]):
        directions[:, z, :, :] = np.moveaxis(star_dist(vol[z], n_directions, opencl=OPENCL_AVAILABLE), -1, 0)

    return directions


# sdist = transforms.Lambda(lambda x: np.moveaxis(star_dist(x[0], N_DIRECTIONS, opencl=OPENCL_AVAILABLE), -1, 0))
sdist = transforms.Lambda(lambda x: sdist_volume(x, N_DIRECTIONS))


labeltotarget = transforms.Compose([sdist])
imagetransform = transforms.Lambda(lambda x: x[None])

train_images = HDF5VolumeLoader(path='data/train-volume.h5', path_in_h5_dataset='data',
                                transforms=imagetransform, **yaml2dict('config_train.yml')['slicing_config'])
train_labels = HDF5VolumeLoader(path='data/labeled_segmentation.h5', path_in_h5_dataset='data',
                                transforms=labeltotarget, **yaml2dict('config_train.yml')['slicing_config'])
trainset = Zip(train_images, train_labels)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCHSIZE,
                                          shuffle=True, num_workers=2)


val_images = HDF5VolumeLoader(path='data/val-volume.h5', path_in_h5_dataset='data',
                              **yaml2dict('config_train.yml')['slicing_config'])
val_labels = HDF5VolumeLoader(path='data/labeled_segmentation_validation.h5', path_in_h5_dataset='data',
                              transforms=labeltotarget, **yaml2dict('config_train.yml')['slicing_config'])
valset = Zip(val_images, val_labels)
valloader = torch.utils.data.DataLoader(trainset, batch_size=BATCHSIZE,
                                        shuffle=True, num_workers=2)


class MyUNet(UNet):
    def __init__(self, in_channels=3, out_channels=1, dim=3, final_activation='ReLU'):
        super(MyUNet, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                     dim=dim, final_activation=final_activation)

    def downsample_op_factory(self, index):
        C = nn.MaxPool2d if self.dim == 2 else nn.MaxPool3d
        return C(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def upsample_op_factory(self, index):
        return nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')

    def _check_scaling(self, input):
        pass


net = torch.nn.Sequential(
    ConvReLU3D(in_channels=1, out_channels=3, kernel_size=3),
    MyUNet(in_channels=3, out_channels=N_DIRECTIONS, dim=3, final_activation='ReLU')
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

print('finished training')

print('the end')
