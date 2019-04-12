import numpy as np

from inferno.io.transform import Transform
from quantizedVDT.utils.affinitiy_utils import get_offset_locations
from stardist import star_dist


class DirectionsToAffinities(Transform):  # not functional atm, do not use

    def __init__(self,  n_directions=8, z_direction=False):
        raise NotImplementedError('remove this error if function is functional')
        super().__init__()
        self.n_directions = n_directions
        self.default_distances = [1, 3, 9, 27]
        self.default_z_distances = [1, 2, 3, 4]
        self.z_direction = z_direction
        self.offsets = []
        if self.z_direction:
            self.offsets += [[-1, 0, 0], [-2, 0, 0], [-3, 0, 0], [-4, 0, 0]]
            self.offsets += [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]
        for i in range(self.n_directions):
            angle = 2*np.pi/self.n_directions*i
            self.offsets += get_offset_locations(self.default_distances, angle)

    def batch_function(self, tensors):
        prediction, target = tensors





class LabelToDirections(Transform):
    def __init__(self, n_directions=8, compute_z=False, opencl_available=True):
        super().__init__()
        self.n_directions = n_directions
        self.opencl_available = opencl_available
        self.compute_z = compute_z


    def batch_function(self, tensors):
        prediction, target = tensors

        if self.compute_z:
            distances = np.empty((self.n_directions+2, *target.shape), dtype=np.float32)
            distances[2:] = sdist_volume(target, self.n_directions,
                                         opencl_available=self.opencl_available)
            distances[:2] = z_dist(target)
        else:
            distances = sdist_volume(target, self.n_directions,
                                     opencl_available=self.opencl_available)
        return prediction, distances



def sdist_volume(vol, n_directions, opencl_available=True):
    """
    returns the n-distances
    :param opencl_available:
    :param n_directions: number of directions
    :param vol: np-like 3d (z,y,x)

    :return: (n_directions, z, y, x)
    """
    directions = np.empty((n_directions, *vol.shape), dtype=np.float32)
    for z in range(vol.shape[0]):
        directions[:, z, :, :] = np.moveaxis(
            star_dist(vol[z], n_directions, opencl=opencl_available), -1, 0)

    return directions


def z_dist(vol):
    """
    Compute the distance to the next change in label in positive and negative z-direction
    :param vol: Volume of image labels
    :return: 4d-Array with shape (2, *vol) that holds the distances
    """
    distances = np.zeros((2, *vol.shape), dtype=np.float32)
    zeros_2d = np.zeros(vol.shape[1:], dtype=np.float32)
    for z in range(1, vol.shape[0]):
        distances[0, z] = np.where(vol[z] == vol[z-1], distances[0, z-1]+1, zeros_2d)
    for z in range(vol.shape[0]-2, -1, -1):
        distances[1, z] = np.where(vol[z] == vol[z+1], distances[1, z+1]+1, zeros_2d)
    return distances


def distancetoaffinities(distance, offsets): #Work in Progress
    raise NotImplementedError
    # return 1.0 if the distance is smaller than the offset, 0.0 if not.
    return [float(distance <= offset) for offset in offsets]




print('halt, stop!')

