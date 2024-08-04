#%%
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import pickle
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer

import torch
from data.NomDataset import ImageCropDataModule
from model.NomResnet import NomResnet101

from torchvision import transforms

from pytorch_lightning import seed_everything
seed_everything(42)

#%%
opt = dict(
    name='ToK1871_SR-HAT',
    max_epochs=100,
    model_mode='train',
    model_ckpt='Backup/pretrained_model/PytorchResNet101Pretrained-data-v2-epoch=14-val_loss_epoch=1.42927-train_acc_epoch=0.99997-val_acc_epoch=0.79039_old.ckpt',
    data_dirs=dict(
        train=[
            'E:/Datasets/TempResources/ToK1871/ToK1871_mixedSR_crops_HAT_wBackground', 
            'E:/Datasets/TempResources/ToK1871/ToK1871_mixedSR_crops.txt'
        ],
        val=[
            'E:/Datasets/TempResources/ToK1902/ToK1902_SR_crops/HAT_SRx2',
            'E:/Datasets/TempResources/ToK1902/ToK1902_crops.txt'
        ],
        test=[
            'E:/Datasets/TempResources/LVT/LVT_SR_raw_crops/HAT_SRx2',
            'E:/Datasets/TempResources/LVT/LVT_crops.txt'
        ]
    ),
    ucode_dict_path='E:/Datasets/NomDataset/HWDB1.1-bitmap64-ucode-hannom-v2-tst_seen-label-set-ucode.pkl',
    input_size=(224, 224),
    transforms=transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(45),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)

#%%
datamodule = ImageCropDataModule(data_dirs=opt['data_dirs'], ucode_dict_path=opt['ucode_dict_path'], batch_size=32, input_size=opt['input_size'], num_workers=0, transforms=opt['transforms'])
n_labels = len(datamodule.ucode_dict.keys())
print(f"Number of labels: {n_labels}")
#%%
dict_table = {}
for k,v in datamodule.ucode_dict.items():
    dict_table[v] = k
print(dict_table)
datamodule.setup('fit')
#%%
sample = next(iter(datamodule.train_dataloader()))
print(sample[0].shape, sample[1].shape)

from matplotlib import pyplot as plt
for i in range(8):
    for j in range(2):
        img = sample[0][i+j].permute(1, 2, 0).numpy()
        plt.subplot(4, 4, i+j+1)
        plt.imshow(img)
        plt.axis('off')
labels = [dict_table[ucode.item()] for ucode in sample[1][:16]]
print('Labels:', labels)

for k in labels:
    if k != 'UNK':
        char = chr(int(k, 16))
        print(char, end=' ')
    else:
        print('UNK', end=' ')

#%%
p = dict(
    max_epochs=50,
    batch_size=32,
    gpus=1,
    num_labels=n_labels,
    model_pretrained_path='../Backup/pretrained_model/PytorchResNet101Pretrained-data-v2-epoch=14-val_loss_epoch=1.42927-train_acc_epoch=0.99997-val_acc_epoch=0.79039_old.ckpt',
    # model_pretrained_path='Checkpoints/Resnet_Ckpt/Resnet_Tok1871_SR-RealESRGANx2_RealCE-epoch=19-train_acc_epoch=0.9931-val_acc_epoch=0.7674-train_loss_epoch=0.0382-val_loss_epoch=1.3757.ckpt',
)


checkpoint_callback = ModelCheckpoint(
    dirpath='Checkpoints/Resnet_HAT_Ckpt/',
    filename='Resnet_Tok1871_SR-HAT-{epoch:02d}-{train_acc_epoch:.4f}-{val_acc_epoch:.4f}-{train_loss_epoch:.4f}-{val_loss_epoch:.4f}',
    save_top_k=1,
    verbose=True,
    monitor='val_acc_epoch',
    mode='max',
    save_last=True
)

early_stopping_callback = EarlyStopping(
    monitor='val_acc_epoch',
    patience=5,
    mode='max'
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = NomResnet101(num_labels=p['num_labels'])
model = NomResnet101.load_from_checkpoint(p['model_pretrained_path'], num_labels=p['num_labels'])
model.to(DEVICE)

#%%
# wandb_logger = WandbLogger(project='SR_Resnet', name="Resnet_SR_ToK1871-RealESRGANx2_mixedSR_wMixedBackground")
# tb_logger = TensorBoardLogger('tensorboard_logs', name='Resnetc_SR_ToK1871-RealESRGANx2_RealCE_wMixedBackground')

 #%%
trainer = Trainer(
        accelerator='gpu',
        max_epochs=p['max_epochs'],
        callbacks=[checkpoint_callback, early_stopping_callback],
        # check_val_every_n_epoch=5
        # logger=[wandb_logger, tb_logger],
    )
trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)

#%%
model = NomResnet101.load_from_checkpoint('Checkpoints/Resnet_HAT_Ckpt/Resnet_Tok1871_SR-HAT_RealCE-epoch=24-train_acc_epoch=0.9963-val_acc_epoch=0.7939-train_loss_epoch=0.0224-val_loss_epoch=1.1749.ckpt', num_labels=p['num_labels'])
model.to(DEVICE)

trainer.test(model, datamodule=datamodule)
# %%
model = NomResnet101.load_from_checkpoint(p['model_pretrained_path'], num_labels=p['num_labels'])
model.to(DEVICE)

trainer.test(model, datamodule=datamodule)
