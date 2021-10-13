from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

mini_normalize = transforms.Normalize(mean=[120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
                                      std=[70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0])
# miniImageNet [大图训练]
transform_A = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomResizedCrop(84),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: x,
        transforms.ToTensor(),
        mini_normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize(92),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        mini_normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize(92),
        transforms.FiveCrop(84),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([mini_normalize(crop) for crop in crops])),
    ])
]

# tieredImageNet [小图训练]
tiered_normalize = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                        np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
transform_B = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # add new
        transforms.RandomHorizontalFlip(),
        lambda x: x,
        transforms.ToTensor(),
        tiered_normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        tiered_normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([tiered_normalize(crop) for crop in crops])),
    ])
]

# FC100 [小图训练]
fc100_normalize = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                       np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
transform_C = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # add new
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize
        fc100_normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        fc100_normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([fc100_normalize(crop) for crop in crops])),
    ])
]

# CIFAR style transformation [小图]
cifar100_normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                          std=[0.2675, 0.2565, 0.2761])
transform_D = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: x,
        transforms.ToTensor(),
        cifar100_normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        cifar100_normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([cifar100_normalize(crop) for crop in crops])),
    ])

]

cub_normalize = transforms.Normalize(mean=np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     std=np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
transform_E = [
    transforms.Compose([
        transforms.RandomResizedCrop(84),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        cub_normalize
        ]),

    transforms.Compose([
        transforms.Resize(92),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        cub_normalize
        ]),
    None
]
transforms_list = ['A', 'B', 'C', 'D', 'E']

transforms_options = {
    'A': transform_A,
    'B': transform_B,
    'C': transform_C,
    'D': transform_D,
    "E": transform_E
}
