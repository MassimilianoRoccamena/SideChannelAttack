from pytorch_lightning import LightningModule

from aidenv.api.model import CoreModel

class DeepModel(LightningModule, CoreModel):
    ''''
    Abstract trainable deep learning model.
    '''

    def set_learning(self, loss, optimizer, scheduler=None):
        '''
        Set model learning parameters.
        '''
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer
        else:
            return [self.optimizer], [self.scheduler]

    def mount(self, *args, **kwargs):
        '''
        Post mount the part of the model which is function of something.
        dataset: dataset object
        '''
        raise NotImplementedError

    # step methods

    def step_batch(self, batch):
        '''
        Compute for a batch: prediction, target, loss.
        '''
        x, y = batch
        y_hat = self(x)
        return self.loss(y_hat, y), \
                y.detach(), \
                y_hat.detach()

    def compute_step(self, batch, prefix):
        step_results = self.step_batch(batch)
        outputs = {'loss':step_results[0],'target':step_results[1],'prediction':step_results[2]}
        return outputs

    def training_step(self, batch, batch_index):
        return self.compute_step(batch, 'train')

    def validation_step(self, batch, batch_index):
        return self.compute_step(batch, 'valid')

    def test_step(self, batch, batch_index):
        return self.compute_step(batch, 'test')

    def forward(self, x):
        raise NotImplementedError
