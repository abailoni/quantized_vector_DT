from quantizedVDT.datasets.cremi_directional import get_cremi_loader
from quantizedVDT.datasets.cremi import get_cremi_loader as get_label_loader
from quantizedVDT.transforms import DirectionsToAffinities
from quantizedVDT.utils.core import reorder_and_invert

from segmfriends.utils.config_utils import recursive_dict_update

from speedrun import BaseExperiment

from quantizedVDT.utils.path_utils import get_source_dir

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from affogato.segmentation import compute_mws_segmentation


class TestOnTruth(BaseExperiment):
    def __init__(self, experiment_directory=None, config=None):
        super(TestOnTruth, self).__init__(experiment_directory)
        # Privates
        self._device = None
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        if config is not None:
            self.read_config_file(config)

        self.DEFAULT_DISPATCH = 'train'
        self.auto_setup()

    def build_train_loader(self):
        return get_cremi_loader(recursive_dict_update(self.get('loaders/train'), self.get('loaders/general')))

 #   def build_label_loader(self):
 #       return get_label_loader(recursive_dict_update(self.get('loaders/train'), self.get('loaders/general')))




if __name__ == '__main__':
    print(sys.argv[1])
    config_path = os.path.join(get_source_dir(), 'experiments/cremi/configs')
    experiments_path = os.path.join(get_source_dir(), './runs/cremi/speedrun')

    sys.argv[1] = os.path.join(experiments_path, sys.argv[1])
    if '--inherit' in sys.argv:
        i = sys.argv.index('--inherit') + 1
        if sys.argv[i].endswith(('.yml', '.yaml')):
            sys.argv[i] = os.path.join(config_path, sys.argv[i])
        else:
            sys.argv[i] = os.path.join(experiments_path, sys.argv[i])
    if '--update' in sys.argv:
        i = sys.argv.index('--update') + 1
        sys.argv[i] = os.path.join(config_path, sys.argv[i])
    i = 0
    while True:
        if f'--update{i}' in sys.argv:
            ind = sys.argv.index(f'--update{i}') + 1
            sys.argv[ind] = os.path.join(config_path, sys.argv[ind])
            i += 1
        else:
            break


    cls = TestOnTruth()
    trainloader = cls.build_train_loader()
#    labelloader = cls.build_label_loader()
    data = trainloader.dataset[0][1]
#    groundtruth = labelloader.dataset[0][1]

    plt.switch_backend('Qt5Agg')

    plt.imshow(data[0, 1])
    plt.show()
    pic = trainloader.dataset[0][0]
    plt.imshow(pic[0, 1])
    plt.show()

    np.save('labels_0.npy', data[0])
    np.save('image_0.npy', pic[0])


    n_d = 2  # number of distances per direction
    n_dir = 16

    dirtoaff = DirectionsToAffinities(n_directions=n_dir, z_direction=True)
    aff = dirtoaff.volume_function_beta(data)


    for i in range(n_dir+2):
        fig, axes = plt.subplots(2, 3)
        axes[0, 0].imshow(data[i, 1].numpy())
        axes[0, 0].set_title(f'distance in direction {i}')
        for j in range(n_d):
            jy, jx = (j+1) % 2, (j+1)//2
            axes[jy, jx].imshow(aff[j+n_d*i, 1])
            axes[jy, jx].set_title(f"aff {j+n_d*i}")

    plt.show()
    offsets = dirtoaff.offsets

    atr = 1

    aff, offsets = reorder_and_invert(aff, offsets, atr*n_dir, dist_per_dir=n_d)

    aff_new = np.empty((aff.shape[0]-12, *aff.shape[1:]))
    aff_new[:6] = aff[[0, 1, 2, 6, 10, 14]]
    aff_new[6:] = aff[18:]
    offsets_new = [offsets[i] for i in [0, 1, 2, 6, 10, 14]] + offsets[18:]

    labels = compute_mws_segmentation(aff_new[:], offsets_new[:], number_of_attractive_channels=6,
                                      strides=None, randomize_strides=False)

    X = labels[1]
    numrows, numcols = X.shape


    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if 0 <= col < numcols and 0 <= row < numrows:
            z = X[row, col]
            return f'x={x:.1f}, y={y:.1f}, z={z}'
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)


    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(labels[1], cmap='tab20b')
    axes[0].set_title('segmentation')
    axes[0].format_coord = format_coord
    axes[1].imshow(trainloader.dataset[0][0][0, 1].numpy())
    axes[1].set_title('input')
    plt.show()


    print('hi')

