from neurofire.metrics.arand import ArandErrorFromMWS
import numpy as np
from quantizedVDT.utils.affinitiy_utils import get_offset_locations
from speedrun.log_anywhere import log_embedding, log_image
from quantizedVDT.transforms import DirectionsToAffinities, Reassemble
from quantizedVDT.utils.core import give_index_of_new_order
# FIXME: fix dependency if needed
try:
    from affogato.segmentation import compute_mws_segmentation
except:
    pass
from inferno.extensions.metrics.base import Metric
import torch.nn
from speedrun.log_anywhere import log_scalar, log_image



class ArandFromMWSDistances(ArandErrorFromMWS):

    def __init__(self, n_directions=4, z_direction=False,
                 strides=None, randomize_strides=False, multiply_by=None,
                 **super_kwargs):
        self.multiply_by = multiply_by
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
        if self.multiply_by is not None:
            distances = distances*self.multiply_by

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


class L1fromQuantized(Metric):

    def __init__(self, n_classes, max_distance, n_distances=8, log=True):
        self.n_classes = n_classes
        self.max_distance = max_distance
        self.reassemble = Reassemble(n_classes, max_distance)
        self.n_distances = n_distances
        self.l1 = torch.nn.L1Loss()
        self.log = log

    def forward(self, prediction, target):

        # log_image('pred_in_metric', prediction)
        # log_image('newtest', prediction)
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        if len(prediction.shape) == 5:
            prediction = prediction[0]  # Kills batches, fix when batchsize >1
            target = target[0]

        distances_pred = self.reassemble.tensor_function(prediction)

        distances_target = self.reassemble.tensor_function(np.concatenate(
            (target[self.n_classes*self.n_distances:],
             target[self.n_distances:self.n_classes*self.n_distances])))

        log_image('distances_tar', torch.Tensor(distances_target[None]))
        log_image('distances_pred', torch.Tensor(distances_pred[None]))



        # if self.log:
        #     for i in range(distances_pred.shape[0]):
        #         log_image(f'predicted_distances_dir_{i}', np.pad(distances_pred[i, :1], pad_width=1, mode='symmetric'))
        #         log_image(f'target_distances_dir_{i}', np.pad(distances_target[i, :1], pad_width=1, mode='symmetric'))
        # the padding is used to create three channels for a rgb-image



        return np.mean(np.abs(distances_pred-distances_target))  # this is without masking, which is kinda bad
        # return self.l1.forward(distances_pred, distances_target)





