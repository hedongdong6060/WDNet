import torch
import os
import os
import importlib

import torch
import torch.nn as nn

from models.build import *
from models.Wavelet import WaveletNet
import torch.nn.functional as F
from dataloader.New_PolypDateset import New_PolypDateset
from dataloader.kvasir_instrument import kvasir_instrument_Dataset
from dataloader.Polyp_Benchmark import Polyp_Benchmark_Dataset


def make_dir(cfg):
    pth_save_dir = os.path.join(cfg["MODEL"]["model_names"], cfg["MODEL"]["model_spec"])

    if not cfg["continue_train"]:
        if os.path.exists(
            os.path.join(
                cfg["TRAIN"]["checkpoints_dir"], pth_save_dir, cfg["DATASET"]["name"]
            )
        ) and (
            any(
                file.endswith(".ckpt")
                for file in os.listdir(
                    os.path.join(
                        cfg["TRAIN"]["checkpoints_dir"],
                        pth_save_dir,
                        cfg["DATASET"]["name"],
                    )
                )
            )
        ):
            name_index = 1
            while os.path.exists(
                os.path.join(
                    cfg["TRAIN"]["checkpoints_dir"],
                    pth_save_dir,
                    cfg["DATASET"]["name"] + "_" + str(name_index),
                )
            ) and (
                any(
                    file.endswith(".ckpt")
                    for file in os.listdir(
                        os.path.join(
                            cfg["TRAIN"]["checkpoints_dir"],
                            pth_save_dir,
                            cfg["DATASET"]["name"] + "_" + str(name_index),
                        )
                    )
                )
            ):

                name_index = name_index + 1

            cfg["DATASET"]["name"] = cfg["DATASET"]["name"] + "_" + str(name_index)

    return os.path.join(
        cfg["TRAIN"]["checkpoints_dir"], pth_save_dir, cfg["DATASET"]["name"]
    )


def build_data(cfg):
    if cfg["DATASET"]["dataset"] == "kvasir-instrument":
        train_set = kvasir_instrument_Dataset(
            cfg, root=cfg["DATASET"]["dataroot"], split="train"
        )
        val_set = kvasir_instrument_Dataset(
            cfg, root=cfg["DATASET"]["dataroot"], split="test"
        )
    elif cfg["DATASET"]["dataset"] == "Polyp_Benchmark":
        train_set = Polyp_Benchmark_Dataset(
            cfg, root=cfg["DATASET"]["dataroot"], split="TrainDataset"
        )
        val_set = Polyp_Benchmark_Dataset(
            cfg, root=cfg["DATASET"]["dataroot"], split="TestDataset"
        )
    elif cfg["DATASET"]["dataset"] == "New_PolypDateset":
        train_set = New_PolypDateset(
            cfg,
            images_directory=cfg["DATASET"]["dataroot"],
            annotations_path="train.json",
        )
        val_set = New_PolypDateset(
            cfg,
            images_directory=cfg["DATASET"]["dataroot"],
            annotations_path="test.json",
        )

    return train_set, val_set


def build_model(train_opt, cfg):

    model = seg_network(
        WaveletNet(in_channels=1, num_classes=cfg["DATASET"]["num_labels"]),
        train_opt,
        cfg,
    )

    return model


def get_ckpt_file(directory):
    print(directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ckpt"):
                print(os.path.join(root, file))
                return os.path.join(root, file)
    return None
