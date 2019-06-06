import numpy as np
import matplotlib.pyplot as plt

import os

from stardist import random_label_cmap

from quantizedVDT.transforms import LabelToDirections, DirectionsToAffinities
from quantizedVDT.utils.core import reorder_and_invert, Annotator, exclude_some_short_edges
from affogato.segmentation import compute_mws_segmentation

from inferno.extensions.metrics.arand import ArandScore

from torch import tensor


# +++++++++++++++++++++++Config area++++++++++++++++++++

n_directions = 16  #don`t set over 16, otherwise no Quantisation later
number_of_attractive_channels = 36
default_distances = [1, 2, 9, 11]
default_z_distances = [1, 2, 3, 4]
less_attr = 1

attr_layers = 2

compute_z = True

# ++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++meta-config+++++++++
save = False
plot_slice = 7
# +++++++++++++++++++++++++++++

assert len(default_z_distances) == len(default_distances)

if save:
    counter = np.load('counter.npy')
    os.mkdir(f'test_{counter}')
    np.save('counter.npy', counter+1)

plt.switch_backend("Qt5Agg")


def plotAffinities(affs, title=None, save_dir=None):
    cols = affs.shape[0] // 2
    fig, ax = plt.subplots(2, cols, figsize=(16, 8))
    for idx, slice in enumerate(affs):
        ax = fig.add_subplot(2, cols, idx + 1)
        ax.imshow(slice)
        ax.set_title(str(idx))
        # ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.suptitle(title)
    if save_dir is not None:
        plt.savefig(save_dir, dpi=500)
    #plt.show()


def plotting_helper(volume):
    for i in range(volume.shape[0]):
        plt.figure()
        plt.imshow(volume[i, plot_slice])
    plt.show()


labels = np.load('labels_0.npy')
image = np.load('image_0.npy')


to_dist = LabelToDirections(n_directions=n_directions, compute_z=compute_z)

_, distances = to_dist.batch_function((image, labels))

distances = np.clip(distances, a_min=None, a_max=100)

#distances = np.min(distances, 20)

to_aff = DirectionsToAffinities(n_directions=n_directions, z_direction=compute_z,
                                default_distances=default_distances,
                                default_z_distances=default_z_distances)

affinities = to_aff.volume_function_beta(distances)
offsets = to_aff.offsets


plotting_helper(distances[:10])

affinities, offsets = reorder_and_invert(affinities, offsets,
                                         n_directions*attr_layers,
                                         dist_per_dir=len(default_distances))

print(f'affinities.shape: \t {affinities.shape} \n'
      f'offsets.shape: \t {np.array(offsets).shape}')

affinities_new, offsets_new = exclude_some_short_edges(affinities, offsets, sampling_factor=less_attr,
                                                       n_directions=n_directions*attr_layers, z_dir=compute_z)
print(offsets_new)

labels_new = compute_mws_segmentation(affinities_new, offsets_new,
                                      number_of_attractive_channels=number_of_attractive_channels,
                                      algorithm='kruskal')

#plotting_helper(affinities_new)
#plotAffinities(affinities_new[:8, 1], 'Affinities')

numrows, numcols = labels_new.shape[1:]

def format_maker(z_values):
    numcols, numrows = z_values.shape
    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if 0 <= col < numcols and 0 <= row < numrows:
            z = z_values[row, col]
            return f'x={x:.1f}, y={y:.1f}, z={z}'
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)
    return format_coord


#for i in range(labels_new.shape[0]):

#    fig, ax = plt.subplots(1, 1)
#    ax.imshow(labels_new[i], cmap='tab20b')
#    ax.format_coord = format_maker(labels_new[i])
#plt.show()




fig, axes = plt.subplots(1, 3, figsize=(20, 8))
axes[0].imshow(labels_new[plot_slice], cmap=random_label_cmap())
axes[0].set_title('reconstruction')
axes[0].format_coord = format_maker(labels_new[plot_slice])
axes[1].imshow(labels[plot_slice], cmap=random_label_cmap())
axes[1].set_title('input')


x = [offset[2] for offset in offsets_new]
y = [offset[1] for offset in offsets_new]
#c = ['b']*number_of_attractive_channels + ['r']*(len(offsets_new)-number_of_attractive_channels)


axes[2].scatter(x[:number_of_attractive_channels], y[:number_of_attractive_channels], c='blue', label='attractive')
axes[2].scatter(x[number_of_attractive_channels:], y[number_of_attractive_channels:], c='red', label='repulsive')
axes[2].legend()
axes[2].set_title('attractive and repulsive edges')
fig.tight_layout()

prediction, target = tensor((labels_new)[None, None, :].astype('int64')), tensor((labels)[None, None, :].astype('int64'))

score = ArandScore().forward(prediction, target)

print(f'The ArandScore is: {score}')

plt.annotate(f'arand: {score}', (1, -5))
#plt.savefig('test_on_truth_6.pdf', dpi=1000)
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(labels_new[plot_slice], cmap=random_label_cmap())
plt.title('reconstruction')
if save:
    plt.savefig(f'test_{counter}/reconstruction.png', dpi=600)


plt.figure(figsize=(4, 4))
plt.scatter(x[:number_of_attractive_channels], y[:number_of_attractive_channels], c='blue', label='attractive')
plt.scatter(x[number_of_attractive_channels:], y[number_of_attractive_channels:], c='red', label='repulsive')
plt.legend()

plt.title('attractive and repulsive edges')
if save:
    plt.savefig(f'test_{counter}/offsets')

if save:
    with open(f'test_{counter}/settings.txt', mode='w') as settings:
        settings.write(f'{score} \tarand-score\n')
        settings.write(f'{n_directions} \t\tnumber of directions\n')
        settings.write(f'{number_of_attractive_channels} \t\tnumber of attractive channels\n')
        settings.write(f'{default_distances} \tdistances in xy-directions\n')
        settings.write(f'{default_z_distances} \tdistances in z-direction\n')
        settings.write(f'{less_attr} \t\tfactor of how many fewer attractive edges are considered \n')
        settings.write(f'{attr_layers} \t\thow many distances are attractive \n')
        settings.write(f'{compute_z} \t\tif we consider the z-dimension\n')
        settings.write(f'\n {offsets_new[:number_of_attractive_channels]} \t\t attractive offsets\n')
        settings.write(f'\n {offsets_new[number_of_attractive_channels:]} \t\t repulsive offsets\n')




print('Bye!')