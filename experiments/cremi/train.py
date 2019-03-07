import os
import logging
import argparse
import yaml
import json
import sys
from torch.nn.modules.loss import BCELoss

import vigra

NUM_WORKERS_PER_BATCH = 25
z_window_slice_training = None


from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.trainers.callbacks.scheduling import AutoLR
from inferno.utils.io_utils import yaml2dict
from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore
# Import the different creiterions, we support.
from inferno.extensions.criteria import SorensenDiceLoss
from inferno.io.transform.base import Compose


from neurofire.criteria.loss_wrapper import LossWrapper
from neurofire.criteria.loss_transforms import MaskTransitionToIgnoreLabel, RemoveSegmentationFromTarget, InvertTarget
from neurofire.models import get_model
from neurofire.datasets.cremi.loaders import get_cremi_loaders

# FIXME:

logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_up_training(project_directory,
                    config,
                    data_config,
                    load_pretrained_model,
                    dir_loaded_model=None):
    VALIDATE_EVERY = ('never')
    SAVE_EVERY = (500, 'iterations')

    # Get model
    if load_pretrained_model:
        load_dir = project_directory if dir_loaded_model is None else dir_loaded_model
        model = Trainer().load(from_directory=load_dir,
                               filename='Weights/checkpoint.pytorch').model
    else:
        model_name = "UNet3D"
        model_kwargs = config.get('pretrained_model_kwargs')
        model = get_model(model_name)(**model_kwargs)

    # Unstructed loss:
    affinity_offsets = data_config['offsets']


    # FIXME: reduce no longer available
    loss = SorensenDiceLoss()

    unstructured_loss = LossWrapper(criterion=loss,
                                    transforms=Compose(MaskTransitionToIgnoreLabel(affinity_offsets),
                                                       RemoveSegmentationFromTarget(),
                                                       InvertTarget()))

    # Build trainer and validation metric
    logger.info("Building trainer.")
    smoothness = 0.95


    # ----------
    # TRAINER:
    # ----------
    trainer = Trainer(model)
    trainer.save_every(SAVE_EVERY, to_directory=os.path.join(project_directory, 'Weights'))
    trainer.build_criterion(unstructured_loss)
    trainer.build_optimizer(**config.get('training_optimizer_kwargs'))
    trainer.evaluate_metric_every('never')
    trainer.validate_every(VALIDATE_EVERY, for_num_iterations=2)
    trainer.register_callback(SaveAtBestValidationScore(smoothness=smoothness, verbose=True))
    trainer.register_callback(AutoLR(factor=1.,
                                  patience='100 iterations',
                                  monitor_while='validating',
                                  monitor_momentum=smoothness,
                                  consider_improvement_with_respect_to='previous'))


    # TODO: define metric
    # trainer.build_metric(metric)

    logger.info("Building logger.")
    # Build logger
    tensorboard = TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                    log_images_every=VALIDATE_EVERY).observe_states(
        ['validation_input', 'validation_prediction, validation_target'],
        observe_while='validating'
    )

    trainer.build_logger(tensorboard, log_directory=os.path.join(project_directory, 'Logs'))
    return trainer


def load_checkpoint(project_directory):
    logger.info("Trainer from checkpoint")
    trainer = Trainer().load(from_directory=os.path.join(project_directory, "Weights"))
    return trainer


def training(project_directory,
             train_configuration_file,
             data_configuration_file,
             validation_configuration_file,
             max_training_iters=int(1e5),
             from_checkpoint=False,
             load_pretrained_model=False,
             dir_loaded_model=None):

    logger.info("Loading config from {}.".format(train_configuration_file))
    config = yaml2dict(train_configuration_file)

    logger.info("Loading training data loader from %s." % data_configuration_file)
    # TODO: adapt configs
    train_loader = get_cremi_loaders(data_configuration_file)
    data_config = yaml2dict(data_configuration_file)

    logger.info("Loading validation data loader from %s." % validation_configuration_file)
    validation_loader = get_cremi_loaders(validation_configuration_file)

    # load network and training progress from checkpoint
    if from_checkpoint:
        logger.info("Loading trainer from checkpoint...")
        trainer = load_checkpoint(project_directory)
    else:
        trainer = set_up_training(project_directory,
                                  config,
                                  data_config,
                                  load_pretrained_model,
                                  dir_loaded_model=dir_loaded_model
                                  )

    trainer.set_max_num_iterations(max_training_iters)

    # Bind loader
    logger.info("Binding loaders to trainer.")
    trainer.bind_loader('train', train_loader).bind_loader('validate', validation_loader)

    # Set devices
    if config.get('devices'):
        logger.info("Using devices {}".format(config.get('devices')))
        trainer.cuda(config.get('devices'))

    # Go!
    logger.info("Lift off!")
    trainer.fit()


def make_train_config(train_config_file, offsets, gpus, nb_threads, reload_model=False):
    if not reload_model:
        template = yaml2dict('./configs/trainer_config.yml')
        template['model_kwargs']['out_channels'] = len(offsets)
    else:
        # Reload previous settings:
        template = yaml2dict(train_config_file)
    template['devices'] = gpus
    with open(train_config_file, 'w') as f:
        yaml.dump(template, f)


def make_data_config(data_config_file, offsets, n_batches, max_nb_workers, reload_model=False):
    if not reload_model:
        template_path = './configs/loader_config_train_set.yml'
        template = yaml2dict(template_path)
        template['offsets'] = offsets
    else:
        # Reload previous settings:
        template = yaml2dict(data_config_file)
    template['loader_config']['batch_size'] = n_batches
    num_workers = NUM_WORKERS_PER_BATCH * n_batches
    template['loader_config']['num_workers'] = num_workers if num_workers < max_nb_workers else max_nb_workers

    # Window size:
    default_wind_size = template['slicing_config']['window_size']['A']
    default_wind_size[0] = z_window_slice_training
    for dataset in template['slicing_config']['window_size']:
        template['slicing_config']['window_size'][dataset] = default_wind_size

    with open(data_config_file, 'w') as f:
        yaml.dump(template, f)


def make_validation_config(validation_config_file, offsets, n_batches, max_nb_workers, reload_model=False):
    if not reload_model:
        template_path = './configs/loader_config_val_set.yml'
        template = yaml2dict(template_path)
        template['offsets'] = offsets
    else:
        # Reload previous settings:
        template = yaml2dict(validation_config_file)
    template['loader_config']['batch_size'] = n_batches
    # num_workers = NUM_WORKERS_PER_BATCH * n_batches
    # template['loader_config']['num_workers'] = num_workers if num_workers < max_nb_workers else max_nb_workers
    template['loader_config']['num_workers'] = 3
    with open(validation_config_file, 'w') as f:
        yaml.dump(template, f)

def make_postproc_config(postproc_config_file, nb_threads, reload_model=False):
    if not reload_model:
        template_path = './template_config/post_proc/post_proc_config.yml'
        template = yaml2dict(template_path)
    else:
        # Reload previous settings:
        template = yaml2dict(postproc_config_file)
    template['nb_threads'] = nb_threads
    with open(postproc_config_file, 'w') as f:
        yaml.dump(template, f)


def parse_offsets(offset_file):
    assert os.path.exists(offset_file)
    with open(offset_file, 'r') as f:
        offsets = json.load(f)
    return offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('offset_file', type=str)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--max_nb_workers', type=int, default=int(8))
    parser.add_argument('--max_train_iters', type=int, default=int(1e5))
    parser.add_argument('--nb_threads', default=int(8), type=int)
    parser.add_argument('--load_model', default='False')
    parser.add_argument('--dir_loaded_model', type=str)
    parser.add_argument('--z_window_size_training', default=int(10), type=int)
    parser.add_argument('--from_checkpoint', default='False')

    # FIXME: get current directory
    base_proj_dir = '/net/hciserver03/storage/abailoni/learnedHC/'



    args = parser.parse_args()

    # Set the proper project folder:
    project_directory = os.path.join(base_proj_dir, args.project_directory)
    if not os.path.exists(project_directory):
        os.mkdir(project_directory)

    # We still leave options for varying the offsets
    # to be more flexible later variable
    base_offs_dir = './experiments/offsets'
    offset_file = os.path.join(base_offs_dir, args.offset_file)
    offsets = parse_offsets(offset_file)

    global z_window_slice_training
    z_window_slice_training = args.z_window_size_training

    max_nb_workers = args.max_nb_workers
    nb_threads = args.nb_threads


    # set the proper CUDA_VISIBLE_DEVICES env variables
    gpus = list(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    gpus = list(range(len(gpus)))

    load_model = eval(args.load_model) or eval(args.from_checkpoint)
    dir_loaded_model = args.dir_loaded_model


    train_config = os.path.join(project_directory, 'train_config.yml')
    make_train_config(train_config, offsets, gpus, nb_threads, reload_model=eval(args.from_checkpoint))

    data_config = os.path.join(project_directory, 'data_config.yml')
    make_data_config(data_config, offsets, len(gpus), max_nb_workers, reload_model=eval(args.from_checkpoint))

    validation_config = os.path.join(project_directory, 'validation_config.yml')
    make_validation_config(validation_config, offsets, len(gpus), max_nb_workers, reload_model=eval(args.from_checkpoint))



    training(project_directory,
             train_config,
             data_config,
             validation_config,
             max_training_iters=args.max_train_iters,
             from_checkpoint=eval(args.from_checkpoint),
             load_pretrained_model=load_model,
             dir_loaded_model=dir_loaded_model)


if __name__ == '__main__':
    main()
