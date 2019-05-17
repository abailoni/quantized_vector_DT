from neurofire.metrics.arand import ArandErrorFromMWS
import numpy as np
from quantizedVDT.utils.affinitiy_utils import get_offset_locations
from speedrun.log_anywhere import log_embedding, log_image
from quantizedVDT.transforms import DirectionsToAffinities
from quantizedVDT.utils.core import give_index_of_new_order
from affogato.segmentation import compute_mws_segmentation


class ArandFromMWSDistances(ArandErrorFromMWS):

    def __init__(self, n_directions=4, z_direction=False,
                 strides=None, randomize_strides=False,
                 **super_kwargs):

        self.laststep = None
        self.n_directions = n_directions
        self.default_distances = [1, 2, 9, 11]
        self.default_z_distances = [1, 2, 3, 4]
        self.z_direction = z_direction
        self.offsets = []
        if self.z_direction:
            self.offsets += [[-1, 0, 0], [-2, 0, 0], [-3, 0, 0], [-4, 0, 0]]
            self.offsets += [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]
        for i in range(self.n_directions):
            angle = 2*np.pi/self.n_directions*i
            self.offsets += get_offset_locations(self.default_distances, angle)

        # change the ordering of offsets and later affinities to comply with what the MWS expects
        self.indexlist_reorder = give_index_of_new_order(len(self.offsets), len(self.default_distances))
        self.offsets = [self.offsets[idx] for idx in self.indexlist_reorder ]

        self.number_of_attractive_channels = 36   # FIXME: Quick-and-dirty fix, get proper number of channels and add subsampling


        # FIXME: Hardcoding this is ugly
        self.transformation = DirectionsToAffinities(n_directions=self.n_directions, z_direction=self.z_direction)

        super().__init__(self.offsets, strides=strides, randomize_strides=randomize_strides,
                         **super_kwargs)

    def input_to_segmentation(self, distances):
        # input: Distances in N directions
        # hardcoded that we have 4 affinities per direction
        affinities = np.empty((distances.shape[0], len(self.default_distances)*distances.shape[1], *distances.shape[2:]))


        for batch in range(distances.shape[0]):
            affinities[batch] = self.transformation.volume_function(distances[batch])[self.indexlist_reorder]
            affinities[batch, :self.number_of_attractive_channels] *= -1
            affinities[batch, :self.number_of_attractive_channels] += 1


        return super(ArandFromMWSDistances, self).input_to_segmentation(affinities)

    def _run_mws(self, input_):
        return compute_mws_segmentation(input_, self.offsets,
                                        number_of_attractive_channels=self.number_of_attractive_channels,
                                        strides=self.strides,
                                        randomize_strides=self.randomize_strides)
