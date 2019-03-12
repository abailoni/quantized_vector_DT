import numpy as np

def directed_distances_3d(arr):
    """

    :param arr: 3d-Integer array of values 0 and 1 and shape (l,m,n)
    :return: distances to closest 1 in 6 directions: Integer-array of shape (l,m,n,6)
    """
    shape = arr.shape
    arr_distances = np.empty((*shape, 6))
    target_dim = 0
    for direc in [1, -1]:   # go in both directions
        for dim in range(3):
            arr = np.swapaxes(arr, 2, dim)
            arr_distances = np.swapaxes(arr_distances, 2, dim)
            for l in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    arr_distances[l, j, ::direc, target_dim] = dist_1d(arr[l, j, ::direc])

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