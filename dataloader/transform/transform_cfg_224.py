from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import imgaug as ia
from imgaug import augmenters as iaa

normalize = transforms.Normalize(mean=[120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
                                 std=[70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0])
# miniImageNet
transform_A = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomHorizontalFlip(),
        lambda x: x,
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
]

transforms_list = ['A']

transforms_options = {
    'A': transform_A,
}
