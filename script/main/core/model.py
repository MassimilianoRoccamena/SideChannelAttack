from pytorch_lightning import LightningModule

from main.base.app.config import ConfigObject

class ConfigModel(ConfigObject, LightningModule):
    ''''
    Abstract configurable model
    '''

    def set_learning(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def training_step(self, batch, batch_index):
        x, y = batch
        y_hat = self(x)
        return self.loss(y_hat, y)

    def validation_step(self, batch, batch_index):
        x, y = batch
        y_hat = self(x)
        return self.loss(y_hat, y)

    def test_step(self, batch, batch_index):
        x, y = batch
        y_hat = self(x)
        return self.loss(y_hat, y)

    def configure_optimizers(self):
        return self.optimizer