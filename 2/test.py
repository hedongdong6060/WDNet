import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from argparse import ArgumentParser
import numpy as np

from util.util import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateFinder

import yaml

import warnings

warnings.filterwarnings("ignore")


parser = ArgumentParser()

parser.add_argument(
    "--cfg",
    type=str,
    default="configs/Polyp_Benchmark.yaml",
    help="Configuration file to use",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoints/xxx/xx/Polyp_Benchmark/Polyp_Benchmark.ckpt",
    help="Path to model checkpoint (if None, loads the latest)",
)

train_opt = parser.parse_args()

with open(train_opt.cfg) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

train_opt.isTrain = True

train_opt.save_pth_dir = make_dir(cfg)
train_set, val_set = build_data(cfg)

train_opt.class_list = val_set.CLASSES

data_loader_train = DataLoader(
    train_set,
    batch_size=cfg["TRAIN"]["batch_size"],
    num_workers=cfg["TRAIN"]["num_workers"],
    pin_memory=True,
    shuffle=True,
)

data_loader_val = DataLoader(
    val_set, batch_size=1, num_workers=cfg["TRAIN"]["num_workers"], pin_memory=True
)

train_opt.total_samples = len(data_loader_train)

model = build_model(train_opt, cfg)

trainer = pl.Trainer(
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    logger=False,
    enable_model_summary=False,
)

trainer.test(model=model, dataloaders=data_loader_val, ckpt_path=train_opt.checkpoint)
