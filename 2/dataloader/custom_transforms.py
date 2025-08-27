import torch
import random
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance, ImageChops
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import pywt


class Resize(object):

    def __init__(self, image_size, other_size):
        self.image_size = image_size  # size: (h, w)
        self.other_size = other_size

    def __call__(self, sample):
        low_freq = sample["low_freq"]
        high_freq = sample["high_freq"]
        mask = sample["label"]

        # resize rgb and label
        low_freq = cv2.resize(low_freq, self.other_size, interpolation=cv2.INTER_LINEAR)
        high_freq = cv2.resize(
            high_freq, self.other_size, interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        return {"low_freq": low_freq, "high_freq": high_freq, "label": mask}


class WaveletTransform(object):

    def __init__(self, wavelet="db2", level=1):
        self.wavelet = wavelet
        self.level = level

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        LL, (LH, HL, HH) = pywt.dwt2(img, self.wavelet)

        LL = (LL - LL.min()) / (LL.max() - LL.min())

        LH = (LH - LH.min()) / (LH.max() - LH.min())
        HL = (HL - HL.min()) / (HL.max() - HL.min())
        HH = (HH - HH.min()) / (HH.max() - HH.min())

        merge1 = HH + HL + LH
        merge1 = (merge1 - merge1.min()) / (merge1.max() - merge1.min())

        return {"low_freq": LL, "high_freq": merge1, "label": mask}


class Normalize_tensor(object):

    def __init__(self, Low_mean, Low_std, High_mean, High_std):
        self.Low_mean = np.array(Low_mean)
        self.Low_std = np.array(Low_std)

        self.High_mean = np.array(High_mean)
        self.High_std = np.array(High_std)

    def __call__(self, sample):
        low_freq = sample["low_freq"]
        high_freq = sample["high_freq"]
        mask = sample["label"]

        low_freq = np.array(low_freq)
        high_freq = np.array(high_freq)

        low_freq = low_freq.astype(np.float32)
        low_freq = low_freq - self.Low_mean
        low_freq = low_freq / self.Low_std

        high_freq = high_freq.astype(np.float32)
        high_freq = high_freq - self.High_mean
        high_freq = high_freq / self.High_std

        low_freq = torch.FloatTensor(low_freq)
        high_freq = torch.FloatTensor(high_freq)
        mask = torch.LongTensor(mask)

        return {"low_freq": low_freq, "high_freq": high_freq, "label": mask}


class ToTensor(object):

    def __call__(self, sample):
        low_freq = sample["low_freq"]
        high_freq = sample["high_freq"]
        mask = sample["label"]

        low_freq = np.array(low_freq)
        high_freq = np.array(high_freq)

        low_freq = torch.FloatTensor(low_freq)
        high_freq = torch.FloatTensor(high_freq)
        mask = torch.LongTensor(mask)

        return {"low_freq": low_freq, "high_freq": high_freq, "label": mask}
