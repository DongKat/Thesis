#%%
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import pickle
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from SR_NomDataset import NomDataModule, NomDatasetCrop
from NomResnet import PytorchResNet101
from torchsummary import summary

#%%
wandb_logger = WandbLogger(project='SR_Resnet', log_model="all", save_dir='wandb_logs', name="SR_Resnet_ESRGAN_Nom")
tb_logger = TensorBoardLogger('lightning_logs', name='SR_Resnet_ESRGAN_Nom')


data_dirs = {'train': ('TempResources/ToK1871/SR_ForResNet/ESRGAN_224', 'TempResources/ToK1871/ToK1871_crops.txt'),
             'val': ('TempResources/ToK1902/ToK1902_crops', 'TempResources/ToK1902/ToK1902_crops.txt'),
             'test': ('TempResources/LVT/LVT_crops', 'TempResources/LVT/LVT_crop.txt'),
             'ucode_dict': 'NomDataset/HWDB1.1-bitmap64-ucode-hannom-v2-tst_seen-label-set-ucode.pkl'
}
datamodule = NomDataModule(data_dirs=data_dirs, batch_size=32, input_size=224, num_workers=8)

# Kinda savage, but it works
with open(data_dirs['ucode_dict'], 'rb') as f:
    ucode_dict = pickle.load(f)
    n_labels = len(ucode_dict.keys())

# TRAIN
p = dict(
    max_epochs=100,
    batch_size=32,
    gpus=1,
    num_labels=n_labels,
    input_dim=224,
    model_mode='train', # 'train' or 'test' or 'resume'
    model_path='SR_Resnet_ESRGAN_Nom_checkpoints/last.ckpt'
)
# TEST
# p = dict(
#     max_epochs=100,
#     batch_size=32,
#     gpus=1,
#     num_labels=n_labels,
#     input_dim=224,
#     model_mode='test', # 'train' or 'test' or 'resume'
#     model_path='SR_Resnet_ESRGAN_Nom_checkpoints/last.ckpt'
# )

# RESUME
# p = dict(
#     max_epochs=100,
#     batch_size=32,
#     gpus=1,
#     num_labels=n_labels,
#     input_dim=224,
#     model_mode='resume', # 'train' or 'test' or 'resume'
#     model_path='SR_Resnet_ESRGAN_Nom_checkpoints/last.ckpt'
# )

checkpoint_callback = ModelCheckpoint(
    dirpath='SR_Resnet_ESRGAN_Nom_checkpoints/',
    filename='SR_Resnet_ESRGAN_Nom-{epoch:02d}-{train_acc_epoch:.4f}-{val_acc_epoch:.4f}-{train_loss_epoch:.4f}-{val_loss_epoch:.4f}',
    save_top_k=1,
    verbose=True,
    monitor='val_acc_epoch',
    mode='max',
    save_last=True
    enable_version_counter=True
)

early_stopping_callback = EarlyStopping(
    monitor='val_acc_epoch',
    patience=50,
    mode='max'
)

model = None
trainer = None
#%%

trainer = Trainer(
        accelerator='gpu',
        max_epochs=p['max_epochs'],
        callbacks=[checkpoint_callback, early_stopping_callback],
        progress_bar_refresh_rate=10,
        logger=[wandb_logger, tb_logger]
    )

if p['model_mode'] == 'train':
    model = PytorchResNet101(p['num_labels'])
    summary(model, (3, p['input_dim'], p['input_dim'])) # (3, 224, 224)
    trainer.fit(model, datamodule=datamodule)
    
elif p['model_mode'] == 'test':
    model = PytorchResNet101.load_from_checkpoint(p['model_path'], num_labels=p['num_labels'])
    summary(model, (3, p['input_dim'], p['input_dim'])) # (3, 224, 224)
    model.eval()
    model.freeze()
    trainer.test(model, datamodule=datamodule)

elif p['model_mode'] == 'resume':
    model = PytorchResNet101(num_labels=p['num_labels'])
    summary(model, (3, p['input_dim'], p['input_dim']))
    # TODO: Check if this works
    trainer.fit(model, datamodule=datamodule,ckpt_path='SR_Resnet_ESRGAN_Nom_checkpoints/last.ckpt')
    