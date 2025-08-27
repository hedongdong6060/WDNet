import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from argparse import ArgumentParser
import numpy as np
from dataloader import custom_transforms as tr

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
    "--output_pred",
    type=str,
    default="./predictions",
    help="Path to save output predictions",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="Polyp_Benchmark.ckpt",
    help="Path to model checkpoint (if None, loads the latest)",
)

parser.add_argument(
    "--overlay",
    type=bool,
    default=True,
    help="Whether to generate the overlay image (original image blended with the mask).",
)
args = parser.parse_args()

with open(args.cfg) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

train_set, val_set = build_data(cfg)
args.class_list = val_set.CLASSES

save_path = os.path.join(args.output_pred, cfg["DATASET"]["dataset"])


data_loader_val = DataLoader(
    val_set, batch_size=1, num_workers=cfg["TRAIN"]["num_workers"], pin_memory=True
)


model = build_model(args, cfg)

trainer = pl.Trainer(
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    logger=False,
    enable_model_summary=False,
)

predictions = trainer.predict(
    model, dataloaders=data_loader_val, ckpt_path=args.checkpoint
)

color_map = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
}


for item in predictions:
    pred = item["pred"].cpu().numpy()
    img_path = item["img_path"][0]

    norm_path = os.path.normpath(img_path)
    parts = norm_path.split(os.sep)
    file_name = parts[-1]
    base_name, _ = os.path.splitext(file_name)

    if len(parts) >= 2:
        subfolder = (
            parts[-3]
            if len(parts) >= 3
            and "Dataset" not in parts[-3]
            and "image" not in parts[-3]
            else ""
        )
    else:
        parent_dir = ""

    if subfolder:
        save_dir = os.path.join(save_path, subfolder)
    else:
        save_dir = save_path

    if "New_PolypDataset" in cfg["DATASET"]["name"]:
        color_mask = np.zeros(
            (cfg["TRAIN"]["size"][0], cfg["TRAIN"]["size"][1], 3), dtype=np.uint8
        )
        for class_id, color in color_map.items():
            color_mask[pred == class_id] = color
        mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    else:
        pred[pred == 1] = 255
        pred = pred.astype(np.uint8)

        mask_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

    not_overlay_dir = os.path.join(save_dir, "not_overlay")
    os.makedirs(not_overlay_dir, exist_ok=True)
    not_overlay_filename = os.path.join(not_overlay_dir, f"{base_name}_pred.png")
    cv2.imwrite(not_overlay_filename, mask_bgr)

    if args.overlay:
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        orig_img = cv2.resize(
            orig_img, (cfg["TRAIN"]["size"][1], cfg["TRAIN"]["size"][0])
        )

        overlay = cv2.addWeighted(orig_img, 0.6, mask_bgr, 0.4, 0)

        overlay_dir = os.path.join(save_dir, "overlay")
        os.makedirs(overlay_dir, exist_ok=True)
        overlay_filename = os.path.join(overlay_dir, f"{base_name}_pred.png")
        cv2.imwrite(overlay_filename, overlay)


print(f"Output predictions saved to {save_path}")
