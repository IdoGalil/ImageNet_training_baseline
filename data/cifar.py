import torch

import torchvision.transforms as T
from torchvision.datasets import CIFAR100

from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule as CIFAR10BoltsDataModule


class CIFAR10DataModule(CIFAR10BoltsDataModule):
    def __init__(self, *args, **kwargs):
        super(CIFAR10DataModule, self).__init__(*args, **kwargs,
                                                train_transforms=T.Compose([
                                                    T.RandomCrop(32, padding=4),
                                                    T.RandomHorizontalFlip(),
                                                    T.ToTensor(),
                                                    cifar10_normalization(),
                                                ]),
                                                val_transforms=T.Compose([
                                                    T.ToTensor(),
                                                    cifar10_normalization(),
                                                ]),
                                                test_transforms=T.Compose([
                                                    T.ToTensor(),
                                                    cifar10_normalization(),
                                                ]))


class CIFAR100DataModule(CIFAR10DataModule):
    name = "cifar100"
    dataset_cls = CIFAR100

    @property
    def num_classes(self):
        return 100
