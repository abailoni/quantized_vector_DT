import numpy as np

from inferno.io.transform import Transform
from quantizedVDT.utils.affinitiy_utils import get_offset_locations
from stardist import star_dist


class DirectionsToAffinities(Transform):  # TODO: Rework

    def __init__(self,  n_directions=8, z_direction=False,
                 default_distances=[1, 3,  8, 12], default_z_distances=[1, 3, 8, 12]):
        super().__init__()

        assert len(default_distances) == len(default_z_distances)

        self.n_directions = n_directions
        self.default_distances = default_distances  # [1, 3, 9, 27]
        self.default_z_distances = default_z_distances  # [1, 2, 3, 4]
        self.z_direction = z_direction
        self.offsets = []
        if self.z_direction:
            self.offsets += [[-d, 0, 0] for d in default_z_distances]
            self.offsets += [[d, 0, 0] for d in default_z_distances]
        for i in range(self.n_directions):
            angle = 2*np.pi/self.n_directions*i
            self.offsets += get_offset_locations(self.default_distances, angle)

    def volume_function(self, distances):

        affinities = np.empty((4*distances.shape[0], *distances.shape[1:]))

        k = 0
        if self.z_direction:
            for i, z_distance in enumerate(self.default_z_distances):
                affinities[i + k * 4, :, :, :] = np.where(distances[k] < z_distance, 1, 0)
            k += 1
            for i, z_distance in enumerate(self.default_z_distances):
                affinities[i + k * 4, :, :, :] = np.where(distances[k] < z_distance, 1, 0)
            k += 1

        while k < distances.shape[0]:
            for i, xy_distance in enumerate(self.default_distances):
                affinities[i + k * 4] = np.where(distances[k] < xy_distance, 1, 0)
            k += 1
        return affinities

    def volume_function_beta(self, distances):
        # will one day be a better way to compute the affinities and replace the current volume_function
        """

        :param distances: array of shape (number of directions, z, y, x)
        :return: affinities: array of shape (number of offsets, z, y, x)
        """
        nr_distances = len(self.default_distances)

        affinities = np.empty((nr_distances*distances.shape[0], *distances.shape[1:]))

        k = 0
        if self.z_direction:
            for i, z_distance in enumerate(self.default_z_distances):
                affinities[i + k * nr_distances, :, :, :] = sigmoid(z_distance, distances[k])
            k += 1
            for i, z_distance in enumerate(self.default_z_distances):
                affinities[i + k * nr_distances, :, :, :] = sigmoid(z_distance, distances[k])
            k += 1

        while k < distances.shape[0]:
            for i, xy_distance in enumerate(self.default_distances):
                affinities[i + k * nr_distances] = sigmoid(xy_distance, distances[k])
            k += 1

        return affinities


class Clip(Transform):

    def __init__(self, a_min=None, a_max=None):
        super().__init__()
        self.a_min = a_min
        self.a_max = a_max

    def volume_function(self, distances):
        return np.clip(distances, a_min=self.a_min, a_max=self.a_max)




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




def sigmoid(x, mean=1, width=None):
    if width is None:
        width = 1
        #width = np.sqrt(mean)/2
    return 1/(1+np.exp((mean-x)/width))





