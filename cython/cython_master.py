import time
import numpy as np
import vigra
import h5py
import matplotlib.pyplot as plt
import dist

set = h5py.File('/export/home/abailoni/datasets/cremi/SOA_affinities/sampleB_train.h5')

groundtruth = set['segmentations/groundtruth_fixed']

example = groundtruth[:, :, :]
example_edges = np.empty_like(example)
for layer in range(example.shape[0]):
    example_layer = vigra.Image(example[layer])
    example_edges[layer] = vigra.analysis.shenCastanEdgeImage(example_layer, 1., 0.5, 1)
    print(f'finished layer {layer}')


def directed_distances_3d(arr):
    shape = arr.shape
    arr_distances = np.empty((*shape, 6))
    target_dim = 0
    for dir in [1, -1]:
        for dim in range(3):
            arr = np.swapaxes(arr, 2, dim)
            arr_distances = np.swapaxes(arr_distances, 2, dim)
            for l in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    arr_distances[l, j, ::dir, target_dim] = dist_1d(arr[l, j, ::dir])

            arr_distances = np.swapaxes(arr_distances, 2, dim)
            arr = np.swapaxes(arr, dim, 2)
            target_dim += 1
    return arr_distances


def dist_1d(arr):
    dist_count = 0
    dist_arr = np.empty_like(arr)
    for i in range(len(arr)):
        if arr[i] == 0:
            dist_count += 1
        else:
            dist_count = 0
        dist_arr[i] = dist_count
    return dist_arr


print('starting here')
start_time = time.time()
our_example = dist.directed_distances_3d(example_edges)
print("--- %s seconds ---" % (time.time() - start_time))

print(our_example.shape)

if True:
    for i in [5]:  # [2,3,4,5,6,7,8]:
        plt.figure()
        plt.imshow(example[i])
        plt.title(f'z = {i}')

    for i in range(6):
        plt.figure()
        plt.imshow(our_example[5, :, :, i])
        print('============')
        print(f'===  i={i} ===')
        print('============')
    plt.show()
