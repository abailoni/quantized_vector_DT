import numpy as np

from inferno.io.transform import Transform

from stardist import star_dist


class DirectionsToAffinities(Transform):  # not functional atm, do not use

    def __init__(self,  n_directions=8, z_direction=False):
        super().__init__()
        self.n_directions = n_directions
        self.default_offsets = [1, 3, 9, 27]
        self.default_z_offsets = [1, 2, 3, 4]
        self.z_direction = z_direction

    def batch_function(self, tensors):
        prediction, target = tensors

        if self.z_direction:
            affinities = np.empty(())
            for i in range(2):
                pass



class LabelToDirections(Transform):
    def __init__(self, n_directions=8, compute_z=False, opencl_available=True):
        super().__init__()
        self.n_directions = n_directions
        self.opencl_available = opencl_available
        self.compute_z = compute_z


    '''
    def volume_function(self, tensor):
        output = sdist_volume(tensor, self.n_directions,
                              opencl_available=self.opencl_available)
        return output
    '''

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
    """
    def batch_function(self, tensors):
        prediction, target = tensors
        target = np.moveaxis(star_dist(target[0].numpy(), self.n_directions, opencl=self.opencl_available), -1, 0)
        return prediction, target
    """


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

    distances = np.zeros((2, *vol.shape), dtype=np.float32)
    zeros_2d = np.zeros(vol.shape[1:], dtype=np.float32)
    for z in range(1, vol.shape[0]):
        distances[0, z] = np.where(vol[z] == vol[z-1], distances[0, z-1]+1, zeros_2d)
    for z in range(vol.shape[0]-2, -1, -1):
        distances[1, z] = np.where(vol[z] == vol[z+1], distances[1, z+1]+1, zeros_2d)
    return distances


def distancetoaffinities(distance, offsets): #Work in Progress
    # return 1.0 if the distance is smaller than the offset, 0.0 if not.
    return [float(distance <= offset) for offset in offsets]

