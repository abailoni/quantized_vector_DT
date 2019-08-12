
from numba import jit
from numba.typed import Dict
from numba import types
import numpy as np

try:
    from affogato.segmentation import compute_mws_segmentation
except:
    print('import affogato failed')
from stardist import ray_angles


# Function made by Ashis
def getFastIOUMWS(dist, S, angles, strides=None, S_attr=1, mask=None, smws=False, verbose=False):
    getOffset = lambda theta, s: list(map(int, list(map(np.around, [s * np.sin(theta), s * np.cos(theta)]))))
    S = float(S)
    S_attr = float(S_attr)

    @jit(nopython=True)
    def validateCoordinates(row, col):
        max_r = dist.shape[1]
        max_c = dist.shape[2]
        in_r = 0 <= row < max_r
        in_c = 0 <= col < max_c
        return in_r & in_c

    @jit(nopython=True)
    def getIntersection(r1: float, _r1: float, r2: float, _r2: float, S: float) -> float:
        inter = 0
        if r1 > S:
            if _r2 > S:
                inter += S
                inter += min(_r1, _r2 - S)
            else:
                inter += _r2
            inter += min(r2, r1 - S)
        else:
            if _r2 > S:
                inter += r1
                inter += min(_r1, _r2 - S)
            else:
                inter += max(r1 + _r2 - S, 0)
        return float(inter)

    @jit(nopython=True, parallel=True)
    def fillAffinities(dist, offset, ray_idx, S):
        rows = dist.shape[1]
        cols = dist.shape[2]
        affs = np.zeros((rows, cols))
        for row in range(rows):
            for col in range(cols):
                row_off = row + offset[0]
                col_off = col + offset[1]
                if validateCoordinates(row_off, col_off):
                    dists_per_pixel0 = dist[:, row, col]
                    dists_per_pixel1 = dist[:, row_off, col_off]
                    if not np.any(dists_per_pixel0) and not np.any(dists_per_pixel1):
                        affs[row, col] = 1.0
                    else:
                        _dist1 = dists_per_pixel0[ray_idx]
                        _dist2 = dists_per_pixel0[ray_idx + n_rays // 2]
                        dist1 = dists_per_pixel1[ray_idx]
                        dist2 = dists_per_pixel1[ray_idx + n_rays // 2]
                        intersection = getIntersection(_dist1.item(), _dist2.item(), dist1.item(), dist2.item(), S)
                        union = _dist1.item() + _dist2.item() + dist1.item() + dist2.item() - intersection
                        if union == 0:
                            iou = 0
                        else:
                            iou = intersection / union
                        affs[row, col] = iou
        return affs

    attractive_angles = [0, 90]
    n_rays = angles.shape[0]
    angl = angles[:n_rays // 2]
    offsets0 = [getOffset(np.deg2rad(angle), S_attr) for angle in attractive_angles]
    offsets1 = [getOffset(a, S) for a in angl]
    offsets = offsets0 + offsets1
    print('Generated offsets:', offsets)
    affs_attr = np.ones((len(offsets0), dist.shape[1], dist.shape[2]))
    angles_d = np.rad2deg(angles)
    off = Dict.empty(key_type=types.int16, value_type=types.int16)
    for idx, offset in enumerate(offsets0):
        ray = np.rad2deg(np.arctan2(*offset))
        ray_idx = np.where(angles_d == ray)[0]
        if verbose: print('Angles', angles_d[ray_idx], angles_d[ray_idx + n_rays // 2])
        off[0] = offset[0]
        off[1] = offset[1]
        affs_attr[idx] = fillAffinities(dist, off, ray_idx[0], S_attr)

    affs_repul = np.zeros((len(offsets1), dist.shape[1], dist.shape[2]))
    for idx, offset in enumerate(offsets1):
        if verbose: print('Angles', np.rad2deg(angles[idx]), np.rad2deg(angles[idx + n_rays // 2]))
        off[0] = offset[0]
        off[1] = offset[1]
        affs_repul[idx] = fillAffinities(dist, off, idx, S)

    merged_aff = np.vstack((affs_attr, affs_repul))
    merged_aff[merged_aff > 1] = 1
    merged_aff[merged_aff < 0] = 0
    merged_aff[len(attractive_angles):] *= -1
    merged_aff[len(attractive_angles):] += 1
    # for i in range(merged_aff.shape[0]):
    #     plt.figure()
    #     plt.imshow(merged_aff[i])
    # plt.show()
    # if smws:
    #     mask = np.expand_dims(mask, 0)
    #     merged_aff = np.vstack((merged_aff, mask))
    #     merged_aff = np.vstack((merged_aff, 1 - mask))
    #     labels = computeSMWS(merged_aff, offsets, len(attractive_angles), stride=strides)
    # else:
    labels = compute_mws_segmentation(merged_aff, offsets, len(attractive_angles), algorithm='kruskal',
                                      strides=strides, mask=mask)

    return labels


def volume_ioumws(dist, S, n_angles):
    angles = ray_angles(n_angles)
    labels = np.zeros(dist.shape[1:], dtype='uint64')
    for z in range(dist.shape[2]):
        labels[z] = getFastIOUMWS(dist[:, z], S, angles)

    return labels


