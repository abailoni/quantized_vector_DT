import numpy as np


def get_offset_locations(offset_distances, angle):
    """

    :param offset_distances: list of scalar distances of offsets
    :param angle: angle in the xy-plane. 0 points in x-direction and turns clockwise
    :return: list of 3d-coordinates of points in offset_distances at angle
    """
    loc = []
    for dist in offset_distances:
        loc += [[0, -1*dist*np.sin(angle), dist*np.cos(angle)]]
    return loc


