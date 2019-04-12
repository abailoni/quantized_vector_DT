from neurofire.metrics.arand import ArandErrorFromMWS
import numpy as np
from quantizedVDT.utils.affinitiy_utils import get_offset_locations


class ArandFromMWSDistances(ArandErrorFromMWS):

    def __init__(self, n_directions=4, z_direction=False,
                 strides=None, randomize_strides=False,
                 **super_kwargs):

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
        super().__init__(self.offsets, strides=strides, randomize_strides=randomize_strides,
                         **super_kwargs)

    def input_to_segmentation(self, distances):  # pray before testing
        # input: Distances in N directions
        # TODO: turn distances into affinities
        # hardcode that we have 4 affinities per direction
        affinities = np.empty((4*distances.shape[0], *distances.shape[1:]))
        k = 0
        if self.z_direction:
            for i, z_distance in enumerate(self.default_z_distances):
                affinities[i+k*4, :, :, :] = np.where(distances[k] <= z_distance, 0, 1)
            k += 1
            for i, z_distance in enumerate(self.default_z_distances):
                affinities[i + k * 4, :, :, :] = np.where(distances[k] <= z_distance, 0, 1)
            k += 1

        while k < distances.shape[0]:
            for i, xy_distance in self.default_distances:
                affinities[i+k*4] = np.where(distances[k] <= xy_distance, 0, 1)
            k += 1

#        affinities =
        return super(ArandFromMWSDistances, self).input_to_segmentation(affinities)
