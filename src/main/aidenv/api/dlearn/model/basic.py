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

    def step_batch(self, batch):
        '''
        Compute (predicted,target,loss) on a batch.
        '''
        x, y = batch
        y_hat = self(x)
        return y_hat.detach().cpu(), \
                y.detach().cpu(), \
                self.loss(y_hat, y)

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer
        else:
            return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_index):
        return self.step(batch)[-1]

    def validation_step(self, batch, batch_index):
        return self.step(batch)[-1]

    def test_step(self, batch, batch_index):
        return self.step(batch)[-1]

    def mount(self, *args, **kwargs):
        '''
        Post mount the part of the model which is function of something.
        dataset: dataset object
        '''
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
