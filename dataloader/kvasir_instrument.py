import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
import cv2
import warnings
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataloader import custom_transforms as tr


class kvasir_instrument_Dataset(data.Dataset):

    CLASSES = ["background", "instrument"]

    def __init__(self, cfg, root="", split="train"):

        self.root = root
        self.split = split
        self.cfg = cfg
        self.images = {}
        self.labels = {}

        self.image_base = os.path.join(self.root, "images", self.split)
        self.label_base = os.path.join(self.root, "masks", self.split)

        self.images[split] = []
        self.images[split] = self.recursive_glob(rootdir=self.image_base, suffix=".jpg")
        self.images[split].sort()

        self.labels[split] = []
        self.labels[split] = self.recursive_glob(rootdir=self.label_base, suffix=".png")
        self.labels[split].sort()

        if not self.images[split]:
            raise Exception(
                "No RGB images for split=[%s] found in %s" % (split, self.image_base)
            )

        if not self.labels[split]:
            raise Exception(
                "No labels for split=[%s] found in %s" % (split, self.label_base)
            )

        print("Found %d %s RGB images" % (len(self.images[split]), split))
        print("Found %d %s label images" % (len(self.labels[split]), split))

    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, index):
        img_path = self.images[self.split][index].rstrip()
        lbl_path = self.labels[self.split][index].rstrip()

        rgb_image = cv2.imread(img_path)
        label_image = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        oriHeight, oriWidth = label_image.shape

        label_image[label_image != 255] = 0
        label_image[label_image == 255] = 1

        sample = {"image": rgb_image, "label": label_image}

        self.path = img_path

        if self.split == "train":
            sample = self.transform_tr(sample)
        elif self.split == "val":
            sample = self.transform_val(sample)
        elif self.split == "test":
            sample = self.transform_ts(sample)
        else:
            sample = self.transform_ts(sample)

        sample["img_path"] = img_path
        sample["oriHeight"] = oriHeight
        sample["oriWidth"] = oriWidth
        sample["oriSize"] = (oriHeight, oriWidth)

        sample["name"] = "/".join(img_path.rsplit("/", 2)[1:])

        return sample

    def recursive_glob(self, rootdir=".", suffix=""):
        """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
        """
        return [
            os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if filename.endswith(suffix)
        ]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.WaveletTransform(),
                tr.Resize(
                    image_size=(
                        self.cfg["TRAIN"]["size"][0],
                        self.cfg["TRAIN"]["size"][1],
                    ),
                    other_size=(128, 128),
                ),
                tr.Normalize_tensor(
                    Low_mean=self.cfg["DATASET"]["Low_mean"],
                    Low_std=self.cfg["DATASET"]["Low_std"],
                    High_mean=self.cfg["DATASET"]["High_mean"],
                    High_std=self.cfg["DATASET"]["High_std"],
                ),
            ]
        )
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.WaveletTransform(),
                tr.Resize(
                    image_size=(
                        self.cfg["TRAIN"]["size"][0],
                        self.cfg["TRAIN"]["size"][1],
                    ),
                    other_size=(128, 128),
                ),
                tr.Normalize_tensor(
                    Low_mean=self.cfg["DATASET"]["Low_mean"],
                    Low_std=self.cfg["DATASET"]["Low_std"],
                    High_mean=self.cfg["DATASET"]["High_mean"],
                    High_std=self.cfg["DATASET"]["High_std"],
                ),
            ]
        )
        return composed_transforms(sample)

    def transform_ts(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.WaveletTransform(),
                tr.Resize(
                    image_size=(
                        self.cfg["TRAIN"]["size"][0],
                        self.cfg["TRAIN"]["size"][1],
                    ),
                    other_size=(128, 128),
                ),
                tr.Normalize_tensor(
                    Low_mean=self.cfg["DATASET"]["Low_mean"],
                    Low_std=self.cfg["DATASET"]["Low_std"],
                    High_mean=self.cfg["DATASET"]["High_mean"],
                    High_std=self.cfg["DATASET"]["High_std"],
                ),
            ]
        )
        return composed_transforms(sample)
