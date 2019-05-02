from neurofire.metrics.arand import ArandErrorFromMWS
import numpy as np
from quantizedVDT.utils.affinitiy_utils import get_offset_locations
from speedrun.log_anywhere import log_embedding, log_image
from quantizedVDT.transforms import DirectionsToAffinities


class ArandFromMWSDistances(ArandErrorFromMWS):

    def __init__(self, n_directions=4, z_direction=False,
                 strides=None, randomize_strides=False,
                 **super_kwargs):

        self.laststep = None
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

        # FIXME: Hardcoding this is ugly
        self.transformation = DirectionsToAffinities(n_directions=self.n_directions, z_direction=self.z_direction)

        super().__init__(self.offsets, strides=strides, randomize_strides=randomize_strides,
                         **super_kwargs)

    def input_to_segmentation(self, distances):
        # input: Distances in N directions
        # TODO: turn distances into affinities
        # hardcoded that we have 4 affinities per direction
        affinities = np.empty((distances.shape[0], 4*distances.shape[1], *distances.shape[2:]))

        for batch in range(distances.shape[0]):
            affinities[batch] = self.transformation.volume_function_beta(distances[batch])

        return super(ArandFromMWSDistances, self).input_to_segmentation(affinities)
