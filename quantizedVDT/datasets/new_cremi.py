from copy import deepcopy

from inferno.io.core import ZipReject, Concatenate
from inferno.io.transform import Compose
from inferno.io.transform.generic import AsTorchBatch
from inferno.io.transform.volume import RandomFlip3D, VolumeAsymmetricCrop
from inferno.io.transform.image import RandomRotate, ElasticTransform
from inferno.utils.io_utils import yaml2dict

from torch.utils.data.dataloader import DataLoader

from neurofire.datasets.loader import RawVolume, SegmentationVolume, RawVolumeWithDefectAugmentation
from neurofire.transform.affinities import affinity_config_to_transform
from neurofire.transform.artifact_source import RejectNonZeroThreshold
from neurofire.transform.volume import RandomSlide

from quantizedVDT.transforms import LabelToDirections, Clip, Multiply


class CremiDataset(ZipReject):
    def __init__(self, name, volume_config, slicing_config,
                 defect_augmentation_config, master_config=None):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)
        assert isinstance(defect_augmentation_config, dict)
        assert 'raw' in volume_config
        assert 'segmentation' in volume_config

        # Get kwargs for raw volume
        raw_volume_kwargs = dict(volume_config.get('raw'))

        # check if we have special dict entries for names in the defect augmentation
        # slicing config
        augmentation_config = deepcopy(defect_augmentation_config)
        for slicing_key, slicing_item in augmentation_config['artifact_source']['slicing_config'].items():
            if isinstance(slicing_item, dict):
                new_item = augmentation_config['artifact_source']['slicing_config'][slicing_key][name]
                augmentation_config['artifact_source']['slicing_config'][slicing_key] = new_item

        raw_volume_kwargs.update({'defect_augmentation_config': augmentation_config})
        raw_volume_kwargs.update(slicing_config)
        # Build raw volume
        self.raw_volume = RawVolumeWithDefectAugmentation(name=name, **raw_volume_kwargs)

        # Get kwargs for segmentation volume
        segmentation_volume_kwargs = dict(volume_config.get('segmentation'))
        segmentation_volume_kwargs.update(slicing_config)
        self.affinity_config = segmentation_volume_kwargs.pop('affinity_config', None)
        # Build segmentation volume
        self.segmentation_volume = SegmentationVolume(name=name,
                                                      **segmentation_volume_kwargs)

        rejection_threshold = volume_config.get('rejection_threshold', 0.6)
        super().__init__(self.raw_volume, self.segmentation_volume,
                         sync=True, rejection_dataset_indices=1,
                         rejection_criterion=RejectNonZeroThreshold(rejection_threshold))
        # Set master config (for transforms)
        self.master_config = {} if master_config is None else master_config
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose()

        for transform in self.master_config:
            transforms.add(transform)

        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        name = config.get('dataset_name')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        defect_augmentation_config = config.get('defect_augmentation_config')
        master_config = config.get('master_config')
        return cls(name, volume_config=volume_config,
                   slicing_config=slicing_config,
                   defect_augmentation_config=defect_augmentation_config,
                   master_config=master_config)


class CremiDatasets(Concatenate):
    def __init__(self, names,
                 volume_config,
                 slicing_config,
                 defect_augmentation_config,
                 master_config=None):
        # Make datasets and concatenate
        if names is None:
            datasets = [CremiDataset(name=None,
                                     volume_config=volume_config,
                                     slicing_config=slicing_config,
                                     defect_augmentation_config=defect_augmentation_config,
                                     master_config=master_config)]
        else:
            datasets = [CremiDataset(name=name,
                                     volume_config=volume_config,
                                     slicing_config=slicing_config,
                                     defect_augmentation_config=defect_augmentation_config,
                                     master_config=master_config)
                        for name in names]
        super().__init__(*datasets)
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = AsTorchBatch(3)
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        names = config.get('names')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        defect_augmentation_config = config.get('defect_augmentation_config')
        master_config = config.get('master_config')
        return cls(names=names, volume_config=volume_config,
                   defect_augmentation_config=defect_augmentation_config,
                   slicing_config=slicing_config, master_config=master_config)


def get_cremi_loader(config):
    """
    Get Cremi loader given a the path to a configuration file.

    Parameters
    ----------
    config : str or dict
        (Path to) Data configuration.

    Returns
    -------
    torch.utils.data.dataloader.DataLoader
        Data loader built as configured.
    """
    config = yaml2dict(config)
    loader_config = config.pop('loader_config')
    datasets = CremiDatasets.from_config(config)
    loader = DataLoader(datasets, **loader_config)
    return loader
