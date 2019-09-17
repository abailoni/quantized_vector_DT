import h5py
import numpy as np
from quantizedVDT.transforms import Reassemble
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("qt5agg")
from neurofire.datasets.loader import SegmentationVolume, RawVolume

segmentation_volume_kwargs = {'dtype': 'float32',
                              'path':
                                  {'A': '/export/home/abailoni/datasets/cremi/SOA_affinities/sampleA_train.h5',
                                   'B': '/export/home/abailoni/datasets/cremi/SOA_affinities/sampleB_train.h5',
                                   'C': '/export/home/abailoni/datasets/cremi/SOA_affinities/sampleC_train.h5'},
                              'path_in_file':
                                  {'A': 'segmentations/groundtruth_fixed',
                                   'B': 'segmentations/groundtruth_fixed',
                                   'C': 'segmentations/groundtruth_fixed'},
                              'data_slice':
                                  {'A': ':, :, :',
                                   'B': '20:32, 300:948, 300:948',
                                   'C': ':75, :, :'},
                              'stride':
                                  {'A': [4, 128, 128],
                                   'B': [4, 128, 128],
                                   'C': [4, 128, 128]},
                              'window_size': {'A': [12, 648, 648],
                                              'B': [12, 648, 648],
                                              'C': [12, 648, 648]}}

raw_volume_kwargs = {'dtype': 'float32',
                              'path':
                                  {'A': '/export/home/abailoni/datasets/cremi/SOA_affinities/sampleA_train.h5',
                                   'B': '/export/home/abailoni/datasets/cremi/SOA_affinities/sampleB_train.h5',
                                   'C': '/export/home/abailoni/datasets/cremi/SOA_affinities/sampleC_train.h5'},
                              'path_in_file':
                                  {'A': 'raw',
                                   'B': 'raw',
                                   'C': 'raw'},
                              'data_slice':
                                  {'A': ':, :, :',
                                   'B': '20:32, 300:948, 300:948',
                                   'C': ':75, :, :'},
                              'stride':
                                  {'A': [4, 128, 128],
                                   'B': [4, 128, 128],
                                   'C': [4, 128, 128]},
                              'window_size': {'A': [12, 648, 648],
                                              'B': [12, 648, 648],
                                              'C': [12, 648, 648]}}

segmentation = SegmentationVolume(name='B', **segmentation_volume_kwargs)[0]

raw = RawVolume(name='B', **raw_volume_kwargs)[0]


f = h5py.File('groundtruth.hdf5', 'a')
for key in f.keys():
    del f[key]
f.create_dataset('raw', data=raw)
f.create_dataset('segmentation', data=segmentation)

f.close()
