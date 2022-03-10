from data.imagenet import ImagenetDataModule
from data.tiny_imagenet import TinyImagenetDataModule
from data.cifar import CIFAR10DataModule
from data.cifar import CIFAR100DataModule

all_data_modules = {
    'imagenet': ImagenetDataModule,
    'tiny_imagenet': TinyImagenetDataModule,
    'cifar10': CIFAR10DataModule,
    'cifar100': CIFAR100DataModule
}