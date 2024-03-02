# Torch libraries imports
import torch
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

# Pytorch Lightning libraries imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Torch utilities libraries imports
import torchmetrics
from torchsummary import summary
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.models import resnet101
from torchvision import transforms


class PytorchResNet101(pl.LightningModule):
    def __init__(self, num_labels):
        super(PytorchResNet101, self).__init__()
        self.save_hyperparameters()
        self.num_labels = num_labels

        # get ResNet architecture
        backbone = resnet101(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Linear(num_filters, self.num_labels)

        self.criterion = nn.CrossEntropyLoss(reduction = 'mean')
        self.metrics_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_labels)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        classification = self.classifier(representations)

        return classification


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
        x, y = batch
        y_hat = self(x)

        # accuracy
        y_hat_softmax = softmax(y_hat)
        y_hat_argmax = torch.argmax(y_hat_softmax, dim=1)

        acc = self.metrics_accuracy(y_hat_argmax, y)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # loss
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.training_step_outputs.append(loss)

        return {'loss': loss, 'train_accuracy': acc}


    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()

        self.log('train_acc_epoch', epoch_average, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs.clear()


    # def convert_to_one_hot(self, y):
    #     vector = np.zeros((y.shape[0], self.num_labels))
    #     vector = torch.eye(self.num_labels)[y]
    #     return torch.tensor(vector, dtype=torch.int)

    # VALIDATION
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # accuracy
        y_hat_softmax = softmax(y_hat)
        y_hat_argmax = torch.argmax(y_hat_softmax, dim=1)

        acc = self.metrics_accuracy(y_hat_argmax, y)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # loss
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.validation_step_outputs.append(loss)

        return {'val_loss': loss, 'val_accuracy': acc}

    def validation_step_end(self, batch_parts):
        return batch_parts

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()

        self.log('val_acc_epoch', epoch_average, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()


    # Saver
    def save_metrics(self, folder, filename, content):
        with open(folder  + filename + '.txt', 'w', encoding='utf-8') as the_txt_file:
            the_txt_file.write(str(content))
        with open(folder + filename + '.pickle', 'wb') as f:
            pickle.dump(content, f)


    # TEST
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        y_hat_softmax = softmax(y_hat)
        y_hat_argmax = torch.argmax(y_hat_softmax, dim=1)

        # accuracy
        acc = self.metrics_accuracy(y_hat_argmax, y)
        self.log('test_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # loss
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.test_step_outputs.append(loss, acc)

        # return {'test_loss': loss, 'test_accuracy': acc, 'ap': ap, 'auroc': auroc, 'confusion_matrix':confmat, 'f1': f1}
        return {'test_loss': loss, 'test_accuracy': acc, 'y': y, 'y_hat':y_hat}

    def test_step_end(self, batch_parts):
        return batch_parts

    def on_test_epoch_end(self, test_step_outputs):
        acc = 0
        loss = 0
        count = 0
        y = None
        y_hat = None
        for test_step_out in test_step_outputs:
            if count == 0:
                y = test_step_out['y']
                y_hat = test_step_out['y_hat']
            else:
                y = torch.cat([y, test_step_out['y']], 0)
                y_hat = torch.cat([y_hat, test_step_out['y_hat']], 0)
            loss += test_step_out['test_loss']
            acc += test_step_out['test_accuracy']
            count += 1

        acc = acc / count
        self.log('test_acc_epoch', acc, on_epoch=True, prog_bar=True, logger=True)
        self.save_metrics('metrics/', 'test_acc_epoch', acc)

        loss = loss / count
        self.log('test_loss_epoch', loss, on_epoch=True, prog_bar=True, logger=True)
        self.save_metrics('metrics/', 'test_loss_epoch', loss)

        y_hat_softmax = softmax(y_hat)
        y_hat_argmax = torch.argmax(y_hat_softmax, dim=1)
        self.save_metrics('ys/', 'y', y)
        self.save_metrics('ys/', 'y_hat', y_hat)
        self.save_metrics('ys/', 'y_hat_softmax', y_hat_softmax)
        self.save_metrics('ys/', 'y_hat_argmax', y_hat_argmax)
        
if __name__ == '__main__':
    seed_everything(42)
    model = PytorchResNet101(num_labels=2)
    print(model)
