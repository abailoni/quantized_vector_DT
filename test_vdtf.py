import vigra
import h5py
import numpy as np

file = h5py.File('/export/home/abailoni/datasets/cremi/SOA_affinities/sampleB_train.h5')

groundtruth = file['segmentations/groundtruth_fixed'].value

x = vigra.ScalarVolume(groundtruth)

x = vigra.ScalarVolume((10, 1650, 1650))



#y = vigra.filters.vectorDistanceTransform(x)
y = vigra.analysis.shenCastanEdgeImage(x[0], 1., 0.5, 1)



print('something')
