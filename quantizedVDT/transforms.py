import numpy as np

from inferno.io.transform import Transform
from quantizedVDT.utils.affinitiy_utils import get_offset_locations
from stardist import star_dist
from keras.utils import to_categorical
from numba import jit
import torch

# class QuantizeDirections(Transform):
#
#     def __init__(self, n_classes):
#         self.n_classes = n_classes



class HomogenousQuantization(Transform):

    def __init__(self, n_classes, max_distance, one_hot=True, apply_to=None):
        # make sure values don't exceed max_distance, otherwise error later
        super().__init__(apply_to=apply_to)
        self.n_classes = n_classes
        self.max_distance = max_distance
        self.classsize = (max_distance-0.001)/(n_classes-1)  # -eps is needed so that largest class is populated
        self.classes = np.arange(n_classes-1, dtype=float)*self.classsize
        self.one_hot = one_hot

    def batch_function(self, tensors):
        prediction, distances = tensors

        classes_shape = (self.n_classes*distances.shape[0], *distances.shape[1:])
        # we don't need the residual of the furthest class
        #residuals = np.empty(((self.n_classes-1)*distances.shape[0], *distances.shape[1:]))

        classidx = np.floor_divide(distances, self.classsize)
        # if self.one_hot:
        ones = np.moveaxis(to_categorical(classidx, num_classes=self.n_classes), -1, 1
                                  ).reshape(classes_shape, order='C')
        # else:
        classes = classidx

        residuals = (distances[None, :]-self.classes.reshape([self.n_classes-1]+[1]*len(distances.shape))
                     ).reshape(((self.n_classes-1)*distances.shape[0], *distances.shape[1:]), order='C').astype(np.float32)

        # residuals = torch.tensor(residuals, dtype=torch.float32)
        # prediction = torch.from_numpy(prediction)
        # classes = torch.from_numpy(classes)

        # return prediction, torch.cat((classes, residuals), dim=0)
        return prediction, np.concatenate((classes, residuals, ones), axis=0)

    def volume_function_jit(self, distances):
        return quantizer(distances, self.n_classes, self.classsize, self.classes)


@jit
def quantizer(distances, n_classes, classsize, classlist):
    classes = np.empty((n_classes*distances.shape[0], *distances.shape[1:]))
    # we don't need the residual of the furthest class
    #residuals = np.empty(((self.n_classes-1)*distances.shape[0], *distances.shape[1:]))

    classidx = np.floor_divide(distances, classsize)
    classes = np.moveaxis(to_categorical(classidx, num_classes=n_classes), -1, 1
                          ).reshape(classes.shape, order='C')

    residuals = (distances[None, :]-classlist.reshape([n_classes-1]+[1]*len(distances.shape))
                 ).reshape(((n_classes-1)*distances.shape[0], *distances.shape[1:]), order='C')
    return np.concatenate((classes, residuals), axis=0)


class Reassemble(Transform):

    def __init__(self, n_classes, max_distance, one_hot=True, apply_to=None):
        # make sure values don't exceed max_distance, otherwise error later
        super().__init__(apply_to=apply_to)
        self.n_classes = n_classes
        self.max_distance = max_distance
        self.classsize = max_distance/(n_classes-1)
        self.classes = np.arange(n_classes-1, dtype=float)*self.classsize
        self.one_hot = one_hot


    def tensor_function(self, values):
        if self.one_hot:
            distances_shape_0 = values.shape[0]//(self.n_classes+(self.n_classes-1))
            classes = values[:self.n_classes*distances_shape_0]
            residuals = np.zeros((self.n_classes*distances_shape_0, *values.shape[1:]))  # classes and residuals should now have the same shape
            residuals[:-1*distances_shape_0] = values[self.n_classes*distances_shape_0:]*self.max_distance


            classidx = np.argmax(classes.reshape([distances_shape_0, self.n_classes, *classes.shape[1:]]), axis=1)
        else:
            distances_shape_0 = (values.shape[0])//self.n_classes
            classidx = values[:distances_shape_0].astype(int)
            residuals = np.zeros((self.n_classes*distances_shape_0, *values.shape[1:]))  # classes and residuals should now have the same shape
            residuals[:-1*distances_shape_0] = values[distances_shape_0:]*self.max_distance

        distance = np.empty_like(classidx, dtype=float)

        # I want to put the distance back together. The step function is easily reconstructed by multiplying
        # the class index by the stepsize. Adding the 'right' residual is more tricky.
        # There are blocks of shape distance.shape lying consecutively for each class. By going to
        # classidx[i]*distances_shape_0 you reach the right block, after which you pick the right element using i[0].

        # This function needs to be made faster, either by using numpy-magic or with cython or similar. (jit?)
        # for i in np.ndindex(classidx.shape):
        #     distance[i] = classidx[i]*self.classsize + \
        #                    residuals[(i[0]+classidx[i]*distances_shape_0,)+i[1:]]

        distanceassembler(distance, classidx, residuals, self.classsize, distances_shape_0)

        # # Pseudocode for how to pick the value that matters from classindex
        # for i in np.ndindex(classidx.shape):
        #     realdistance = alldistances[(classidx[i],)+i]

        return distance


@jit
def distanceassembler(distance, classidx, residuals, classsize, distances_shape_0):
    for i in np.ndindex(classidx.shape):
            distance[i] = classidx[i]*classsize + \
                           residuals[(i[0]+classidx[i]*distances_shape_0,)+i[1:]]




class DirectionsToAffinities(Transform):  # TODO: Rework

    def __init__(self,  n_directions=8, z_direction=False,
                 default_distances=[1, 3,  8, 12], default_z_distances=[1, 3, 8, 12], apply_to=None):
        super().__init__(apply_to=apply_to)

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


class Mask(Transform):

    def __init__(self, n_dir, a_max, apply_to=None):
        super().__init__(apply_to=apply_to)
        self.n_dir = n_dir
        self.a_max = a_max

    def volume_function(self, distances):
        mask = np.ones(distances.shape)
        for i in range(self.n_dir):
            xoffset = -int(np.cos(i / self.n_dir * 2 * np.pi) * self.a_max)
            yoffset = -int(np.sin(i / self.n_dir * 2 * np.pi) * self.a_max)
            xslice = slice(xoffset - 1, None) if xoffset < 0 else slice(None, xoffset + 1)
            yslice = slice(yoffset - 1, None) if yoffset < 0 else slice(None, yoffset + 1)
            mask[i, :, yslice, :] = 0
            mask[i, :, :, xslice] = 0
        mask = torch.Tensor(mask).cuda()
        return distances*mask


class Clip(Transform):

    def __init__(self, a_min=None, a_max=None, apply_to=None):
        super().__init__(apply_to=apply_to)
        self.a_min = a_min
        self.a_max = a_max

    def volume_function(self, distances):
        return np.clip(distances, a_min=self.a_min, a_max=self.a_max)


class Multiply(Transform):

    def __init__(self, factor, invert_factor=False, apply_to=None):
        super().__init__(apply_to=apply_to)
        self.factor = 1/factor if invert_factor else factor

    def volume_function(self, distances):
        return distances*self.factor


class LabelToDirections(Transform):
    def __init__(self, n_directions=8, compute_z=False, opencl_available=True, apply_to=None):
        super().__init__(apply_to=apply_to)
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







