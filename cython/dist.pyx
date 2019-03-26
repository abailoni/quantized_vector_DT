import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t

def directed_distances_3d(np.ndarray[unsigned int, ndim=3] arr):
    """

    :param arr: 3d-Integer array of values 0 and 1 and shape (l,m,n)
    :return: distances to closest 1 in 6 directions: Integer-array of shape (l,m,n,6)
    """
    cdef int direc, dim, l, j
    cdef np.ndarray[int, ndim=4] arr_distances = np.empty((arr.shape[0], arr.shape[1], arr.shape[2], 6), dtype=np.int32)
    cdef int target_dim = 0
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


def dist_1d(np.ndarray[unsigned int, ndim=1] arr):
    cdef int dist_count = 0
    cdef np.ndarray[unsigned int, ndim=1] dist_arr = np.empty_like(arr)
    cdef int i
    for i in range(len(arr)):
        if arr[i] == 0:
            dist_count += 1
        else:
            dist_count = 0
        dist_arr[i] = dist_count
    return dist_arr





def get_top_left_distance(np.ndarray[unsigned int, ndim=2] arr):
    """
    get the distance to the next 1 in the top left direction
    :param arr: 2d-arr (m, n)
    :return: dist_arr: (m, n)
    """
    cdef int x, y
    cdef np.ndarray[int, ndim=2] dist_arr = np.zeros_like(arr, dtype=np.int32)
    dist_arr[0,:] = 0

    for y in range(1, arr.shape[0]):
        dist_arr[y, 1:] = dist_arr[y-1, :-1] + 1
        for x in range(arr.shape[1]):
            if arr[y, x] == 1:
                dist_arr[y, x] = 0
    return dist_arr