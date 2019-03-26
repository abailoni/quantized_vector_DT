import time
import numpy as np
import vigra
import h5py
import matplotlib.pyplot as plt
import dist

set = h5py.File('/export/home/abailoni/datasets/cremi/SOA_affinities/sampleB_train.h5')

groundtruth = set['segmentations/groundtruth_fixed']

example = groundtruth[100, :, :]
example_edges = np.empty_like(example)
example_edges[:] = vigra.analysis.shenCastanEdgeImage(vigra.Image(example), 1., 0.5, 1)
"""
example_edges = np.empty_like(example)
for layer in range(example.shape[0]):
    example_layer = vigra.Image(example[layer])
    example_edges[layer] = vigra.analysis.shenCastanEdgeImage(example_layer, 1., 0.5, 1)
    print(f'finished layer {layer}')
"""


def get_top_left_distance(arr):
    """
    get the distance to the next 1 in the top left direction
    :param arr: 2d-arr (m, n)
    :return: dist_arr: (m, n)
    """
    dist_arr = np.zeros_like(arr, dtype=np.int32)
    dist_arr[0,:] = 0

    for y in range(1, arr.shape[0]):
        dist_arr[y, 1:] = dist_arr[y-1, :-1] + 1
        for x in range(arr.shape[1]):
            if arr[y, x] == 1:
                dist_arr[y, x] = 0
    return dist_arr


print('starting here')
start_time = time.time()

#our_example = dist.directed_distances_3d(example_edges)
our_example = dist.get_top_left_distance(example_edges)


print("--- %s seconds ---" % (time.time() - start_time))

# plot some views of the data to check correctness
plt.figure()
plt.imshow(example_edges)
plt.figure()
plt.imshow(our_example)
plt.show()


if False:
    for i in [5]:  # [2,3,4,5,6,7,8]:
        plt.figure()
        plt.imshow(example[i])
        plt.title(f'z = {i}')

    for i in range(6):
        plt.figure()
        plt.imshow(our_example[5, :, :, i])

    plt.show()
