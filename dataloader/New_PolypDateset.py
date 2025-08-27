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
from pycocotools.coco import COCO
import random


from dataloader import custom_transforms as tr


class New_PolypDateset(data.Dataset):

    CLASSES = ["background", "Polyp", "instrument"]

    def __init__(self, cfg, images_directory, annotations_path):

        self.cfg = cfg
        self.images = {}
        self.labels = {}

        self.IMAGES_DIRECTORY = os.path.join(
            images_directory, os.path.splitext(annotations_path)[0]
        )
        self.ANNOTATIONS_PATH = os.path.join(images_directory, annotations_path)

        self.coco = COCO(self.ANNOTATIONS_PATH)

        self.image_ids = self.coco.getImgIds(catIds=self.coco.getCatIds())

        print("Found %d RGB images" % (len(self.image_ids)))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):

        idx = self.image_ids[index]

        img = self.coco.loadImgs(idx)[0]
        image_path = os.path.join(self.IMAGES_DIRECTORY, img["file_name"])

        rgb_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        oriHeight, oriWidth = img["height"], img["width"]
        category_mask = np.zeros((oriHeight, oriWidth), dtype=np.uint8)

        annotation_ids = self.coco.getAnnIds(imgIds=img["id"])
        annotations = self.coco.loadAnns(annotation_ids)

        random.shuffle(annotations)

        for ann in annotations:
            mask = self.coco.annToMask(ann)
            category_id = ann["category_id"] + 1
            category_mask[mask == 1] = category_id

        sample = {"image": rgb_image, "label": category_mask}

        self.path = image_path

        if self.ANNOTATIONS_PATH in "train":
            sample = self.transform_tr(sample)
        elif self.ANNOTATIONS_PATH in "val":
            sample = self.transform_val(sample)
        elif self.ANNOTATIONS_PATH in "test":
            sample = self.transform_ts(sample)
        else:
            sample = self.transform_ts(sample)

        sample["img_path"] = image_path
        sample["oriHeight"] = oriHeight
        sample["oriWidth"] = oriWidth
        sample["oriSize"] = (oriHeight, oriWidth)

        sample["name"] = "/".join(image_path.rsplit("/", 2)[1:])

        return sample

    def recursive_glob(self, rootdir=".", suffix=""):
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
