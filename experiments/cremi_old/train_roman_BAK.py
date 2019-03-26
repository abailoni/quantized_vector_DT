from speedrun import BaseExperiment, TensorboardMixin, InfernoMixin
from speedrun.log_anywhere import register_logger, log_image, log_scalar
from speedrun.py_utils import locate

import os
import torch
import torch.nn as nn

from SegTags.models.config_parsing import parse_model
from SegTags.datasets.sfrp.loaders import get_sfrp_loader
from SegTags.callbacks import SaveModelAtBestValidationScore, PrintString
from SegTags.metrics.inferno_metrics import SFRPMetric
from embeddingutils.loss import WeightedLoss, SumLoss
from neurofire.criteria.loss_wrapper import LossWrapper
from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
from inferno.trainers.callbacks import Callback

from shutil import copyfile
import sys

from inferno.trainers.callbacks.metric import ExtraMetric
from inferno.extensions.layers.convolutional import ConvELU3D, Conv3D, BNReLUConv3D

from fancy.dictselect import recursive_update
from fancy.slicing import center_slice


class ReduceDatasetCallback(Callback):
    def end_of_training_iteration(self, **_):
        iteration = self.trainer.iteration_count
        if iteration + 1 == len(self.trainer.train_loader):
            print('Removing rejected samples from train dataset')
            for dset in self.trainer.train_loader.dataset.datasets:
                dset.remove_rejected()


class Maiden(BaseExperiment, InfernoMixin, TensorboardMixin):
    def __init__(self, experiment_directory=None, config=None):
        super(Maiden, self).__init__(experiment_directory)
        # Privates
        self._device = None
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        if config is not None:  # TODO
            self.read_config_file(config)

        self.DEFAULT_DISPATCH = 'train'
        self.auto_setup()

        # register additional callbacks
        self.register_run_info_callback()
        self.register_reduce_dataset_callback()
        self.trainer.register_callback(
            SaveModelAtBestValidationScore(to_directory=self.checkpoint_directory, smoothness=0, verbose=True))
        if self.get('trainer/criterion/losses/fgbg') is not None:
            self.build_fgbg_metric()
        # register anywhere logger for scalars
        register_logger(self, 'scalars')

    def build_fgbg_metric(self):
        self.trainer.register_callback(ExtraMetric(
            LossWrapper(
                SorensenDiceLoss(channelwise=True),
                transforms=self.to_fgbg_loss_input
            ),
            frequency=self.get('trainer/metric/evaluate_every'),
            name='error_semantic_dice'
        ))

    def register_run_info_callback(self):
        msg = f'Run: {self.experiment_directory}'

        class RunInfoCallback(PrintString):
            def print(self):
                print(msg)

        self.trainer.register_callback(RunInfoCallback((25, 'iterations')))

    def register_reduce_dataset_callback(self):
        self.trainer.register_callback(ReduceDatasetCallback())

    def build_model(self, model_config=None):
        model_config = self.get('model') if model_config is None else model_config
        self.build_final_activation(model_config)
        return super(Maiden, self).build_model(model_config) #parse_model(model_config)

    def build_final_activation(self, model_config=None):
        model_config = self.get('model') if model_config is None else model_config
        model_class = list(model_config.keys())[0]
        final_layer = None
        final_activation_config = model_config[model_class].pop('final_activation', True)
        if not final_activation_config:
            return
        elif final_activation_config is True:
            final_activation_config = {}
        for layer_name, kwargs in final_activation_config.items():
            layer_class = locate(
                layer_name, ['embeddingutils.models.submodules', 'inferno.extensions.layers.convolutional'])
            final_layer = layer_class(**kwargs)

        model_config[model_class]['final_activation'] = \
            Conv3D(in_channels=model_config[model_class]['out_channels'],
                   out_channels=model_config[model_class]['out_channels'],
                   kernel_size=3)

        if final_layer is not None:
            model_config[model_class]['final_activation'] = nn.Sequential(
                model_config[model_class]['final_activation'], final_layer)

    @staticmethod
    def to_fgbg_loss_input(prediction, target):
        # return first channel of prediction, foreground mask of target
        if torch.is_tensor(target):
            target = [target]
        return torch.sigmoid(prediction[:, :1]), (target[0] != 0).float()

    def parse_and_wrap_losses(self, config, transforms, losses, weights, loss_names):
        default_weight = config.pop('weight', 1)
        for class_name, kwargs in config.items():
            loss_names.append(kwargs.pop('name', class_name))
            weights.append(kwargs.pop('weight', default_weight))
            print(f'Adding {loss_names[-1]} with weight {weights[-1]}')
            loss_class = locate(class_name,
                                ['embeddingutils.loss',
                                 'SegTags.loss',
                                 'inferno.extensions.criteria.set_similarity_measures',
                                 'torch.nn'])
            if issubclass(loss_class, WeightedLoss):
                kwargs['trainer'] = self.trainer
            losses.append(LossWrapper(
                criterion=loss_class(**kwargs),
                transforms=transforms
            ))

    def inferno_build_criterion(self):
        print("Building criterion")
        loss_config = self.get('trainer/criterion/losses')
        losses, weights, loss_names = [], [], []

        def parse_and_wrap_losses(key, transforms):
            self.parse_and_wrap_losses(loss_config.get(key, dict()), transforms,
                                       losses, weights, loss_names)

        parse_and_wrap_losses('fgbg', transforms=self.to_fgbg_loss_input)

        handcrafted_dim = loss_config.get('handcrafted_tags', dict()).pop('dim', 4)

        def to_tag_loss_input(prediction, target):
            prediction = prediction[:, 1:handcrafted_dim + 1]
            return prediction, [target[0], target[2]]
        parse_and_wrap_losses('handcrafted_tags', transforms=to_tag_loss_input)

        free_start_channel = loss_config.get('free_embedding', dict()).pop('start_channel', 1)

        def to_free_embedding_loss_input(prediction, target):
            return prediction[:, free_start_channel:], target[0]
        parse_and_wrap_losses('free_embedding', transforms=to_free_embedding_loss_input)

        sum_loss_kwargs = dict(
            trainer=self.trainer,
            loss_weights=weights, losses=losses, loss_names=loss_names,
            **self.get('trainer/criterion/sum_loss_kwargs')
        )
        self._trainer.build_criterion(SumLoss(**sum_loss_kwargs))

    def inferno_build_metric(self):
        metric_config = self.get('trainer/metric')
        frequency = metric_config.pop('evaluate_every', (25, 'iterations'))
        self.trainer.evaluate_metric_every(frequency)
        if metric_config:
            assert len(metric_config) == 1
            for class_name, kwargs in metric_config.items():
                cls = locate(class_name)
                if issubclass(cls, SFRPMetric):
                    kwargs['trainer'] = self.trainer
                print(f'Building metric of class "{cls.__name__}"')
                metric = cls(**kwargs)
                if hasattr(self, 'metric_pre'):
                    pre = self.metric_pre()
                    self.trainer.build_metric(lambda prediction, target: metric(*pre(prediction, target)))
                else:
                    self.trainer.build_metric(metric)
        self.set('trainer/metric/evaluate_every', frequency)

    def build_train_loader(self):
        return get_sfrp_loader(recursive_update(self.get('loaders/general'), self.get('loaders/train')))

    def build_val_loader(self):
        return get_sfrp_loader(recursive_update(self.get('loaders/general'), self.get('loaders/val')))


class TwoStageModel(torch.nn.Module):
    def __init__(self, stage_1, stage_2, truncate_grad=True):
        super(TwoStageModel, self).__init__()
        self.stage_1 = stage_1
        self.stage_2 = stage_2
        self.truncate_grad = truncate_grad

    def forward(self, inp_1):
        e1 = self.stage_1(inp_1)
        inp_2 = torch.cat([inp_1[(slice(None),)*2 + center_slice(inp_1.shape[-3:], e1.shape[-3:])], e1], dim=1)
        if self.truncate_grad:
            inp_2.detach_().requires_grad_()
        e2 = self.stage_2(inp_2)
        return [e1, e2]


class TwoStageExperiment(Maiden):
    def build_model(self, model_config=None):
        model_config = self.get('model') if model_config is None else model_config
        stage_1 = super(TwoStageExperiment, self).build_model(model_config.pop('stage_1'))
        stage_2 = super(TwoStageExperiment, self).build_model(model_config.pop('stage_2'))
        return TwoStageModel(stage_1, stage_2, **model_config)

    def inferno_build_criterion(self):
        print("Building criterion")
        loss_config = self.get('trainer/criterion/losses')

        def to_push_loss_input(prediction, target):
            return prediction[0], target

        def to_constrain_loss_input(prediction, target):
            # average e1 over GT segments
            result = []
            for e1, gt_seg in zip(prediction[0], target):
                result_slice = e1.new_zeros(e1.shape)
                gt_seg = gt_seg[0].long()
                ind = gt_seg != 0
                gt_seg = gt_seg[ind] - 1
                e1 = e1[:, ind]
                n_segments = torch.max(gt_seg).int().item() + 1
                centroids = []
                for seg_id in range(n_segments):
                    segment_mask = gt_seg == seg_id
                    centroids.append(e1[:, segment_mask].mean(-1))
                centroids = torch.stack(centroids, dim=-1)
                pixel_wise_centroids = centroids[:, gt_seg]
                result_slice[:, ind] = pixel_wise_centroids
                result.append(result_slice)
            result = torch.stack(result)
            return result, result.new_zeros(result.shape)

        def to_pull_loss_input(prediction, target):
            # average e1 over GT segments
            result = []
            for e1, gt_seg in zip(prediction[0].detach(), target):
                result_slice = e1.new_zeros(e1.shape)
                gt_seg = gt_seg[0].long()
                ind = gt_seg != 0
                gt_seg = gt_seg[ind] - 1
                e1 = e1[:, ind]
                n_segments = torch.max(gt_seg).int().item() + 1
                centroids = []
                for seg_id in range(n_segments):
                    segment_mask = gt_seg == seg_id
                    centroids.append(e1[:, segment_mask].mean(-1))
                centroids = torch.stack(centroids, dim=-1)
                pixel_wise_centroids = centroids[:, gt_seg]
                result_slice[:, ind] = pixel_wise_centroids
                result.append(result_slice)
            return prediction[1], [target, torch.stack(result, dim=0)]

        losses, weights, loss_names = [], [], []
        self.parse_and_wrap_losses(loss_config.get('push_loss'), to_push_loss_input,
                                   losses, weights, loss_names)
        self.parse_and_wrap_losses(loss_config.get('constrain_loss'), to_constrain_loss_input,
                                   losses, weights, loss_names)
        self.parse_and_wrap_losses(loss_config.get('pull_loss'), to_pull_loss_input,
                                   losses, weights, loss_names)
        sum_loss_kwargs = dict(
            trainer=self.trainer,
            loss_weights=weights, losses=losses, loss_names=loss_names,
            **self.get('trainer/criterion/sum_loss_kwargs')
        )
        self._trainer.build_criterion(SumLoss(**sum_loss_kwargs))

    @staticmethod
    def metric_pre():
        return lambda prediction, target: (prediction[-1], target)


if __name__ == '__main__':
    print(sys.argv[1])
    config_path = 'config/speedrun/sfrp'
    experiments_path = 'runs/sfrp/speedrun'

    sys.argv[1] = os.path.join(experiments_path, sys.argv[1])
    if '--inherit' in sys.argv:
        i = sys.argv.index('--inherit') + 1
        if sys.argv[i].endswith(('.yml', '.yaml')):
            sys.argv[i] = os.path.join(config_path, sys.argv[i])
        else:
            sys.argv[i] = os.path.join(experiments_path, sys.argv[i])
    if '--update' in sys.argv:
        i = sys.argv.index('--update') + 1
        sys.argv[i] = os.path.join(config_path, sys.argv[i])
    i = 0
    while True:
        if f'--update{i}' in sys.argv:
            ind = sys.argv.index(f'--update{i}') + 1
            sys.argv[ind] = os.path.join(config_path, sys.argv[ind])
            i += 1
        else:
            break
    cls = TwoStageExperiment if 'two_stage' in sys.argv[1] else Maiden
    cls().run()

    '''

    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
    else:
        folder_name = "dev_00"

    if len(sys.argv) > 2:
        config_name = sys.argv[2]
    else:
        config_name = "sfrp.yml"

    if '/' in config_name:
        config_path = config_name
        config_name = config_path.split('/')[-1]
    else:
        config_path = f"config/speedrun/{config_name}"

    # TODO: use speedrun autosetup

    if 'two_stage' in config_name:
        project_directory = f'runs/sfrp/two_stage/{folder_name}'
    else:
        project_directory = f'runs/sfrp/speedrun/{folder_name}/'

    assert not os.path.exists(project_directory) or 'dev_' in project_directory
    if not os.path.exists(project_directory):
        print("creating new project directory ", project_directory)
        os.mkdir(project_directory)
        print("copying template config")
        os.mkdir(project_directory + "/Configurations/")
        print(f"{project_directory}/Configurations/{config_name}")

    # TODO: this is risky
    copyfile(config_path,
             f"{project_directory}/Configurations/{config_name}")

    cls = TwoStageExperiment if config_name == 'sfrp_two_stage.yml' else Maiden

    exp = cls(project_directory, f"{config_name}")
    exp.train()
    '''
