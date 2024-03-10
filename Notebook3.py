# This notebook aims at fixing SR - Detection
# Train Yolo more with SR images

#%%
from NomDataset import NomDatasetV1, NomDatasetV2

# Standard libraries imports
import importlib
import os
import glob
import random
import shutil
from tqdm.autonotebook import tqdm
import pybboxes as pbx


# Data manipulation libraries imports (for image processing)
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

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

torch.cuda.empty_cache()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")