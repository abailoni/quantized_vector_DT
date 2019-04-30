from neurofire.metrics.arand import ArandErrorFromMWS
import numpy as np
from quantizedVDT.utils.affinitiy_utils import get_offset_locations
from speedrun.log_anywhere import log_embedding, log_image


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
        super().__init__(self.offsets, strides=strides, randomize_strides=randomize_strides,
                         **super_kwargs)

    def input_to_segmentation(self, distances):
        # input: Distances in N directions
        # TODO: turn distances into affinities
        # hardcoded that we have 4 affinities per direction
        affinities = np.empty((distances.shape[0], 4*distances.shape[1], *distances.shape[2:]))
        for batchnum in range(distances.shape[0]):
            k = 0
            if self.z_direction:
                for i, z_distance in enumerate(self.default_z_distances):
                    affinities[batchnum, i+k*4, :, :, :] = np.where(distances[batchnum, k] <= z_distance, 1, 0)
                k += 1
                for i, z_distance in enumerate(self.default_z_distances):
                    affinities[batchnum, i+k*4, :, :, :] = np.where(distances[batchnum, k] <= z_distance, 1, 0)
                k += 1

            while k < distances.shape[1]:
                for i, xy_distance in enumerate(self.default_distances):
                    affinities[batchnum, i+k*4] = np.where(distances[batchnum, k] <= xy_distance, 1, 0)
                k += 1


            #log_embedding('tensor', affinities[0, 0, 1])
            #for i in range(affinities.shape[1]):
            #    log_image(f'affinities/channel{i}', affinities[0, i, 1])
                # not only not pretty but also depends on a modification I made to the logger
                # addition was made in speedrun/tensorboard.py lines 51 and 55
    #        affinities =
        return super(ArandFromMWSDistances, self).input_to_segmentation(affinities)
