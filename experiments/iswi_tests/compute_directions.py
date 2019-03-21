import h5py
import numpy as np
from stardist import star_dist
import matplotlib.pyplot as plt

n_directions = 8

file = h5py.File('gt_cleaned.h5')
data = np.array(file['data'][:, :, :])

distances = np.empty((n_directions, *data.shape), dtype=np.int32)

for z in range(data.shape[0]):
    distances[:, z, :, :] = np.moveaxis(star_dist(data[z], 8), -1, 0)

print(distances.shape)



newfile = h5py.File('stardistance.h5', 'w')
newfile2 = h5py.File('stardistance_val.h5', 'w')

newfile.create_dataset('data', data=distances, dtype=np.float32)
newfile2.create_dataset('data', data=distances, dtype=np.float32)




