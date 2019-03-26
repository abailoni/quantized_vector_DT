import h5py
import matplotlib.pyplot as plt


set = h5py.File('/export/home/abailoni/datasets/cremi/SOA_affinities/sampleB_train.h5')

groundtruth = set['segmentations/groundtruth_fixed']

print(groundtruth[100, 1000:1010, 1000:1010])

plt.imshow(groundtruth[100, :, :])
plt.show()