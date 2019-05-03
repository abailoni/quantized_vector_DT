import numpy as np
import matplotlib.pyplot as plt

from quantizedVDT.transforms import LabelToDirections, DirectionsToAffinities
from quantizedVDT.utils.core import reorder_and_invert, Annotator, exclude_some_short_edges
from affogato.segmentation import compute_mws_segmentation


# +++++++++++++++++++++++Config area++++++++++++++++++++

n_directions = 16
number_of_attractive_channels = n_directions
default_distances = [1, 4]
default_z_distances = [1, 2]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++

plt.switch_backend("Qt5Agg")


def plotting_helper(volume):
    for i in range(volume.shape[0]):
        plt.figure()
        plt.imshow(volume[i, 1])
    plt.show()


labels = np.load('labels_0.npy')
image = np.load('image_0.npy')

to_dist = LabelToDirections(n_directions=n_directions, compute_z=True)

_, distances = to_dist.batch_function((image, labels))

to_aff = DirectionsToAffinities(n_directions=n_directions, z_direction=True,
                                default_distances=default_distances,
                                default_z_distances=default_z_distances)

affinities = to_aff.volume_function_beta(distances)
offsets = to_aff.offsets


#plotting_helper(affinities)

affinities, offsets = reorder_and_invert(affinities, offsets,
                                         number_of_attractive_channels,
                                         dist_per_dir=len(default_distances))

print(f'affinities.shape: \t {affinities.shape} \n'
      f'offsets.shape: \t {np.array(offsets).shape}')

affinities_new, offsets_new = exclude_some_short_edges(affinities, offsets, sampling_factor=4,
                                                       n_directions=n_directions)
print(offsets_new)

labels_new = compute_mws_segmentation(affinities, offsets,
                                      number_of_attractive_channels=6,
                                      strides=None, randomize_strides=False)

plotting_helper(affinities_new)

X = labels_new[1]
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
axes[0].imshow(labels_new[1], cmap='tab20b')
axes[0].set_title('segmentation')
axes[0].format_coord = format_coord
axes[1].imshow(labels[1])
axes[1].set_title('input')
plt.show()

print('Bye!')