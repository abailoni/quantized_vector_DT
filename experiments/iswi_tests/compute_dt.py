import vigra
import h5py
import numpy as np

file = h5py.File('gt_cleaned.h5')
data = np.array(file['data'][:, :, :])
"""
print(data.shape)
boundaries = np.empty_like(data, dtype=np.int32)
for i in range(data.shape[0]):
    boundaries[i] = vigra.analysis.shenCastanEdgeImage(vigra.Image(data[i]), 1., 0.5, 1)
    print(i)
"""

#distances = vigra.filters.boundaryDistanceTransform(data)

distances = np.empty((*data.shape, 2), dtype=np.float32)

for i in range(data.shape[0]):
    edge = vigra.analysis.regionImageToEdgeImage(data[i])
    distances[i] = vigra.filters.vectorDistanceTransform(vigra.Image(edge))




#targetfile = h5py.File('./vector_dist_trans.h5', 'w')
#target_data = targetfile.create_dataset('data', data=distances, dtype=np.float32)
#print(target_data.shape)