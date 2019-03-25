from speedrun import BaseExperiment, TensorboardMixin, InfernoMixin
from speedrun.log_anywhere import register_logger, log_image, log_scalar
from speedrun.py_utils import locate

import os
import torch
import torch.nn as nn

from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore
from neurofire.criteria.loss_wrapper import LossWrapper
from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
from inferno.trainers.callbacks import Callback

from embeddingutils.loss import WeightedLoss, SumLoss

from shutil import copyfile
import sys

from inferno.extensions.layers.convolutional import ConvELU3D, Conv3D, BNReLUConv3D



class ReduceDatasetCallback(Callback):
    def end_of_training_iteration(self, **_):
        iteration = self.trainer.iteration_count
        if iteration + 1 == len(self.trainer.train_loader):
            print('Removing rejected samples from train dataset')
            for dset in self.trainer.train_loader.dataset.datasets:
                dset.remove_rejected()


class BaseCremiExperiment(BaseExperiment, InfernoMixin, TensorboardMixin):
    def __init__(self, experiment_directory=None, config=None):
        super(BaseCremiExperiment, self).__init__(experiment_directory)
        # Privates
        self._device = None
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        if config is not None:  # TODO
            self.read_config_file(config)

        self.DEFAULT_DISPATCH = 'train'
        self.auto_setup()

        # register additional callbacks
        # self.register_run_info_callback()
        self.register_reduce_dataset_callback()



        self.trainer.register_callback(
            SaveAtBestValidationScore(smoothness=0, verbose=True))
            # SaveModelAtBestValidationScore(to_directory=self.checkpoint_directory, smoothness=0, verbose=True))

        # FIXME: wait, this is a second criterion..? The main one is built afterwards
        # if self.get('trainer/criterion/losses/fgbg') is not None:
        #     self.build_fgbg_metric()

        # TODO: understand
        # register anywhere logger for scalars
        register_logger(self, 'scalars')

    # def build_fgbg_metric(self):
    #     self.trainer.register_callback(ExtraMetric(
    #         LossWrapper(
    #             SorensenDiceLoss(channelwise=True),
    #             transforms=self.to_fgbg_loss_input
    #         ),
    #         frequency=self.get('trainer/metric/evaluate_every'),
    #         name='error_semantic_dice'
    #     ))

    # def register_run_info_callback(self):
    #     msg = f'Run: {self.experiment_directory}'
    #
    #     class RunInfoCallback(PrintString):
    #         def print(self):
    #             print(msg)
    #
    #     self.trainer.register_callback(RunInfoCallback((25, 'iterations')))

    def register_reduce_dataset_callback(self):
        self.trainer.register_callback(ReduceDatasetCallback())

    def build_model(self, model_config=None):
        model_config = self.get('model') if model_config is None else model_config
        # self.build_final_activation(model_config)
        return super(BaseCremiExperiment, self).build_model(model_config) #parse_model(model_config)

    # def build_final_activation(self, model_config=None):
    #     model_config = self.get('model') if model_config is None else model_config
    #     model_class = list(model_config.keys())[0]
    #     final_layer = None
    #     final_activation_config = model_config[model_class].pop('final_activation', True)
    #     if not final_activation_config:
    #         return
    #     elif final_activation_config is True:
    #         final_activation_config = {}
    #     for layer_name, kwargs in final_activation_config.items():
    #         layer_class = locate(
    #             layer_name, ['embeddingutils.models.submodules', 'inferno.extensions.layers.convolutional'])
    #         final_layer = layer_class(**kwargs)
    #
    #     model_config[model_class]['final_activation'] = \
    #         Conv3D(in_channels=model_config[model_class]['out_channels'],
    #                out_channels=model_config[model_class]['out_channels'],
    #                kernel_size=3)
    #
    #     if final_layer is not None:
    #         model_config[model_class]['final_activation'] = nn.Sequential(
    #             model_config[model_class]['final_activation'], final_layer)

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
    cls = TwoStageExperiment if 'two_stage' in sys.argv[1] else BaseCremiExperiment
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

    cls = TwoStageExperiment if config_name == 'sfrp_two_stage.yml' else BaseCremiExperiment

    exp = cls(project_directory, f"{config_name}")
    exp.train()
    '''
