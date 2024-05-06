# This code is copied from Thu's work
# Fixed to work with new version Python, Pytorch and Pytorch Lightning
#%%
# Torch libraries imports
import torch
from torch import nn
from torch.nn.functional import softmax
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

# Pytorch Lightning libraries imports
import pytorch_lightning as pl

# Torch utilities libraries imports
import torchmetrics
from torchvision.models import resnet

#%%
class PytorchResNet101(pl.LightningModule):
    def __init__(self, num_labels):
        super(PytorchResNet101, self).__init__()
        self.save_hyperparameters()
        self.num_labels = num_labels

        # Get ResNet architecture and remove the last FC layer
        backbone = resnet.resnet101(weights=resnet.ResNet101_Weights.DEFAULT)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        
        # Initialize layers
        self.feature_extractor = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(num_filters, self.num_labels)

        self.criterion = nn.CrossEntropyLoss()
        self.metrics = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_labels)
    
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=0.00001, weight_decay=1e-4, momentum=0.9)
        # optimizer = Adam(self.parameters(), lr=0.001, eps=1)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.0005)
        # return dict(
        #     optimizer=optimizer,
        #     lr_scheduler=dict(
        #         scheduler=scheduler,
        #         interval='step'
        #     )
        # )
        return optimizer


    def training_step(self, batch, batch_idx):
        x, y = batch # x.shape = (batch_size, 3, 224, 224), y.shape = (batch_size, 1)
        
        # Inference
        y_hat = self(x) # y_hat.shape = (batch_size, num_labels)
        y_hat_softmax = softmax(y_hat, dim=1)
        y_hat_argmax = torch.argmax(y_hat_softmax, dim=1)

        # Accuracy
        acc = self.metrics(y_hat_argmax, y)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Loss
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.training_step_outputs.append((loss, acc))
        
        return loss


    def on_train_epoch_end(self):
        loss_epoch_average = torch.stack([loss for loss, _ in self.validation_step_outputs]).mean()
        acc_epoch_average = torch.stack([acc for _, acc in self.validation_step_outputs]).mean()
        self.log('train_loss_epoch', loss_epoch_average, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_epoch', acc_epoch_average, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs.clear()
        


    # VALIDATION
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # accuracy
        y_hat_softmax = softmax(y_hat)
        y_hat_argmax = torch.argmax(y_hat_softmax, dim=1)

        acc = self.metrics(y_hat_argmax, y)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # loss
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.validation_step_outputs.append((loss, acc))

        return loss


    def on_validation_epoch_end(self):
        loss_epoch_average = torch.stack([loss for loss, _ in self.validation_step_outputs]).mean()
        acc_epoch_average = torch.stack([acc for _, acc in self.validation_step_outputs]).mean()
        self.log('val_loss_epoch', loss_epoch_average, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_epoch', acc_epoch_average, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()


    # TEST
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        y_hat_softmax = softmax(y_hat)
        y_hat_argmax = torch.argmax(y_hat_softmax, dim=1)

        # loss
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # accuracy
        acc = self.metrics(y_hat_argmax, y)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.test_step_outputs.append((loss, acc))
        
        return acc

    def on_test_epoch_end(self):
        loss_epoch_average = torch.stack([loss for loss, _ in self.test_step_outputs]).mean()
        acc_epoch_average = torch.stack([acc for _, acc in self.test_step_outputs]).mean()
        self.log('test_acc_epoch', acc_epoch_average, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_loss_epoch', loss_epoch_average, on_epoch=True, prog_bar=True, logger=True)
        self.test_step_outputs.clear()

