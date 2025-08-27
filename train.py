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

torch.cuda.set_per_process_memory_fraction(1.0, 0)

parser = ArgumentParser()

parser.add_argument(
    "--cfg",
    type=str,
    default="./configs/Polyp_Benchmark.yaml",
    help="Configuration file to use",
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
    drop_last=True,
)

data_loader_val = DataLoader(
    val_set, batch_size=1, num_workers=cfg["TRAIN"]["num_workers"], pin_memory=True
)

checkpoint_callback = ModelCheckpoint(
    monitor="index/average_IoU",
    dirpath=train_opt.save_pth_dir,
    filename=cfg["DATASET"]["name"],
    mode="max",
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")

tensorboard_logger = TensorBoardLogger(save_dir=train_opt.save_pth_dir)
#
train_opt.total_samples = len(data_loader_train)

model = build_model(train_opt, cfg)

trainer = pl.Trainer(
    # strategy=DDPStrategy(),
    devices=cfg["TRAIN"]["node"],
    max_epochs=cfg["TRAIN"]["nepoch"],
    callbacks=[checkpoint_callback, lr_monitor],
    default_root_dir=train_opt.save_pth_dir,
    logger=tensorboard_logger,
    log_every_n_steps=50,
    check_val_every_n_epoch=cfg["TRAIN"]["eval_interval"],
    enable_model_summary=False,
)

if not cfg["continue_train"]:
    trainer.fit(
        model=model,
        train_dataloaders=data_loader_train,
        val_dataloaders=data_loader_val,
    )
else:
    pth_dir = get_ckpt_file(train_opt.save_pth_dir)
    trainer.fit(
        model=model,
        train_dataloaders=data_loader_train,
        val_dataloaders=data_loader_val,
        ckpt_path=pth_dir,
    )
