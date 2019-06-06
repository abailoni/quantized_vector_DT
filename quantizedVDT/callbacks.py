from inferno.trainers.callbacks.base import Callback


class SaveModelCallback(Callback):

    def __init__(self, save_every):
        super(Callback, self).__init__()
        self.save_every = save_every

    def end_of_training_iteration(self, **_):
        if self.trainer._iteration_count % self.save_every == 0:
            self.trainer.save_model()
