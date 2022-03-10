import torch

from pl_bolts.datamodules.imagenet_datamodule import ImagenetDataModule as ImagenetBoltsDataModule
from pl_bolts.datasets.imagenet_dataset import UnlabeledImagenet


class ImagenetDataModule(ImagenetBoltsDataModule):
    def __init__(
            self,
            data_dir,
            meta_dir=None,
            num_imgs_per_val_class=0,
            image_size=224,
            num_workers=0,
            batch_size=32,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            seed=42,
            *args,
            **kwargs,
    ) -> None:
        super(ImagenetDataModule, self).__init__(data_dir=data_dir,
                                                 meta_dir=meta_dir,
                                                 num_imgs_per_val_class=0,
                                                 image_size=image_size,
                                                 num_workers=num_workers,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 pin_memory=pin_memory,
                                                 drop_last=drop_last,
                                                 *args,
                                                 **kwargs)
        self.seed = seed
