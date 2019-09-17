import h5py
import numpy as np
from quantizedVDT.transforms import Reassemble
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("qt5agg")
from neurofire.datasets.loader import SegmentationVolume
from quantizedVDT.transforms import LabelToDirections
from inferno.extensions.metrics.arand import ArandScore
from stardist import random_label_cmap

a_max = 40


def plot(volume, title=None, save=None, show=False):
    if volume.ndim == 4:
        for i in [1]:
            plt.figure()
            plt.imshow(volume[i, 1], vmin=0, vmax=a_max)
            plt.title(title)
            plt.colorbar()
            if save:
                plt.savefig(f'pics/{save}_dir_{title}')
    elif volume.ndim == 3:
        plt.figure()
        plt.imshow(volume[1])
        plt.title(title)
        plt.colorbar()
        if save:
            plt.savefig(f'pics/{save}_dir_{title}')
    else:
        plt.figure()
        plt.imshow(volume, cmap=random_label_cmap(n=1))
        plt.title(title)
        plt.colorbar()
        if save:
            plt.savefig(f'pics/{save}_dir_{title}')
    if show:
        plt.show()

f1 = h5py.File('/export/home/claun/PycharmProjects/quantized_vector_DT/runs/cremi/speedrun/21_8_quant_infer/inference.hdf5', 'r')
outputs1 = f1['outputs'][0]
f1.close()
re = Reassemble(4, a_max)
dirs1 = re(outputs1)

f4 = h5py.File('/export/home/claun/PycharmProjects/quantized_vector_DT/runs/cremi/speedrun/27_8_quant_4_dist_infer/inference.hdf5', 'r')
outputs4 = f4['outputs'][0]
f4.close()
re = Reassemble(4, a_max)
dirs4 = re(outputs4)

f5 = h5py.File('/export/home/claun/PycharmProjects/quantized_vector_DT/runs/cremi/speedrun/27_8_quant_8_class_infer/inference.hdf5', 'r')
outputs5 = f5['outputs'][0]
f5.close()
re = Reassemble(8, a_max)
dirs5 = re(outputs5)

f6 = h5py.File('/export/home/claun/PycharmProjects/quantized_vector_DT/runs/cremi/speedrun/2_9_noquant_4d_infer/inference.hdf5', 'r')
dirs6 = f6['outputs'][0]
f6.close()

f7 = h5py.File('/export/home/claun/PycharmProjects/quantized_vector_DT/runs/cremi/speedrun/11_9_3_class_8_dir_infer/inference.hdf5', 'r')
outputs7 = f7['outputs'][0]
f7.close()
re = Reassemble(3, a_max)
dirs7 = re(outputs7)

f8 = h5py.File('/export/home/claun/PycharmProjects/quantized_vector_DT/runs/cremi/speedrun/11_9_2_class_8_dir_infer/inference.hdf5', 'r')
outputs8 = f8['outputs'][0]
f8.close()
re = Reassemble(2, a_max)
dirs8 = re(outputs8)


# plt.show()


f2 = h5py.File('/export/home/claun/PycharmProjects/quantized_vector_DT/runs/cremi/speedrun/21_8_noquant_infer/inference.hdf5', 'r')

dirs2 = f2['outputs'][0]*a_max
f2.close()

# plt.show()


# segmentation_volume_kwargs = {'dtype': 'float32',
#                               'path':
#                                   {'A': '/export/home/abailoni/datasets/cremi/SOA_affinities/sampleA_train.h5',
#                                    'B': '/export/home/abailoni/datasets/cremi/SOA_affinities/sampleB_train.h5',
#                                    'C': '/export/home/abailoni/datasets/cremi/SOA_affinities/sampleC_train.h5'},
#                               'path_in_file':
#                                   {'A': 'segmentations/groundtruth_fixed',
#                                    'B': 'segmentations/groundtruth_fixed',
#                                    'C': 'segmentations/groundtruth_fixed'},
#                               'data_slice':
#                                   {'A': ':, :, :',
#                                    'B': '20:23, 300:570, 300:570',
#                                    'C': ':75, :, :'},
#                               'stride':
#                                   {'A': [4, 128, 128],
#                                    'B': [4, 128, 128],
#                                    'C': [4, 128, 128]},
#                               'window_size': {'A': [3, 270, 270],
#                                               'B': [3, 270, 270],
#                                               'C': [3, 270, 270]}}

# segmentation = SegmentationVolume(name='B', **segmentation_volume_kwargs)[0]
f3 = h5py.File('groundtruth.hdf5', 'r')
segmentation = f3['segmentation'][:]
raw = f3['raw'][:]
f3.close()
d = LabelToDirections(8)

_, truth = d(5, segmentation)
truth = np.clip(truth, a_min=None, a_max=40)


dirs1 = dirs1[:, 1:-1, 50:-50, 50:-50]
dirs2 = dirs2[:, 1:-1, 50:-50, 50:-50]
dirs4 = dirs4[:, 1:-1, 50:-50, 50:-50]
dirs5 = dirs5[:, 1:-1, 50:-50, 50:-50]
dirs6 = dirs6[:, 1:-1, 50:-50, 50:-50]
dirs7 = dirs7[:, 1:-1, 50:-50, 50:-50]
dirs8 = dirs8[:, 1:-1, 50:-50, 50:-50]
truth = truth[:, 1:-1, 50:-50, 50:-50]
segmentation = segmentation[1:-1, 50:-50, 50:-50]
truth2 = truth[::2]
raw = raw[1:-1, 50:-50, 50:-50]
boundaryidx = np.nonzero(np.where(truth <= 39, 1, 0))

#
# plot(dirs1, save=False, title='quantized')
# plot(dirs2, save=False, title='non_quantized')
# plot(truth, save=False, title='groundtruth')
# diff1 = np.abs(truth-dirs1)
# diff2 = np.abs(truth-dirs2)
# diff4 = np.abs(truth2-dirs4)
# diff5 = np.abs(truth2-dirs5)
#
# plot(diff1, title='diff_quant')
# plot(diff2, title='diff_nonq')
# plt.show()
#
# plt.figure()
# plt.hist(diff1.flatten(), bins=2000, density=True)
# plt.title('quantized')
# # plt.yscale('log')
# # plt.show()
#
# # plt.ylim(ymax=1e7)
#
#
# # plt.figure()
# plt.hist(diff2.flatten(), bins=2000, alpha=0.5, density=True)
# plt.vlines([13.3, 26.6, 39.9], -0.1, 2)
# # plt.title('non_quantized')
# # plt.yscale('log')
# plt.show()
#
plt.figure()
plt.hist2d(dirs1.flatten(), truth.flatten(), bins=[100, 40], norm=matplotlib.colors.LogNorm())
plt.show()

plt.figure()
plt.hist2d(dirs2.flatten(), truth.flatten(), bins=[100, 40], norm=matplotlib.colors.LogNorm())
plt.show()



import sys
sys.path.insert(1, '/export/home/claun/PycharmProjects/quantized_vector_DT/preparation')
from iou_mws import getFastIOUMWS
from stardist import ray_angles
import torch
angl = ray_angles(8)
labels1 = np.array(getFastIOUMWS(dirs1[:, 1], 10, angl), dtype=np.int64)
labels2 = np.array(getFastIOUMWS(dirs2[:, 1], 10, angl), dtype=np.int64)
labels3 = np.array(getFastIOUMWS(truth[:, 1], 10, angl), dtype=np.int64)

# plt.figure()
# plt.imshow(labels1, cmap='tab20')
# plt.title('quantized')
# plt.figure()
# plt.imshow(labels2, cmap='tab20')
# plt.title('non quantized')
# plt.figure()
# plt.imshow(labels3, cmap='tab20')
# plt.title('reconstructed ground truth')
# plt.figure()
# plt.imshow(segmentation[1], cmap='tab20')
# plt.title('segmentation ground truth')
# plt.show()

arand = ArandScore()

a1 = 1 - arand(torch.Tensor(labels1[None, None]), torch.Tensor(segmentation[1])[None, None])
a2 = 1 - arand(torch.Tensor(labels2[None, None]), torch.Tensor(segmentation[1])[None, None])
a3 = 1 - arand(torch.Tensor(labels3[None, None]), torch.Tensor(segmentation[1])[None, None])

print('No gap removal:', a1, a2, a3)

# a1 = []
# a2 = []
# a3 = []


# gg = [-5, -1, 0, 0.01, 0.1, 1]
# hh = [-5, -1, 0, 0.01, 0.1, 1, 5]

gg = np.linspace(-5, 5, 41)
hh = np.linspace(-5, 5, 41)

a1 = np.ones((len(gg), len(hh))) # quantized
a2 = np.ones((len(gg), len(hh))) # not quantized
a3 = np.ones((len(gg), len(hh)))


def dostuff(i, g):
    for j, h in enumerate(hh):
        labels1 = np.array(getFastIOUMWS(dirs1[:, 1], 10, angl, gap_eps_attr=g, gap_eps_rep=h), dtype=np.int64)
        labels2 = np.array(getFastIOUMWS(dirs2[:, 1], 10, angl, gap_eps_attr=g, gap_eps_rep=h), dtype=np.int64)
        labels3 = np.array(getFastIOUMWS(truth[:, 1], 10, angl, gap_eps_attr=g, gap_eps_rep=h), dtype=np.int64)

        a1[i, j] = 1 - arand(torch.Tensor(labels1[None, None]), torch.Tensor(segmentation[1])[None, None])
        a2[i, j] = 1 - arand(torch.Tensor(labels2[None, None]), torch.Tensor(segmentation[1])[None, None])
        a3[i, j] = 1 - arand(torch.Tensor(labels3[None, None]), torch.Tensor(segmentation[1])[None, None])

        print(i, j)


for i, g in enumerate(gg):
    dostuff(i, g)

#
# import multiprocessing as mp
# output = mp.Queue()
# processes = [mp.Process(target=dostuff, args=(i, g)) for i, g in enumerate(gg)]
# for p in processes:
#     p.start()
#
# for p in processes:
#     p.join()

# a[x,y] x is attr, y is rep
plt.figure()
plt.title('quantized')
plt.imshow(a1, extent=[-5, 5, -5, 5], vmin=0, vmax=1)
plt.colorbar()
plt.savefig('params_quant')
# plt.show()
plt.figure()
plt.title('non quantized')
plt.imshow(a2, extent=[-5, 5, -5, 5], vmin=0, vmax=1)
plt.colorbar()
plt.savefig('params_noquant')
# plt.show()
plt.figure()
plt.title('ground truth')
plt.imshow(a3, extent=[-5, 5, -5, 5], vmin=0, vmax=1)
plt.colorbar()
plt.savefig('params_gt')
plt.show()





