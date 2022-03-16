import os
from monai.transforms.compose import Compose
from typing import Optional, Union
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from monai.data import (
    CacheDataset,
    load_decathlon_datalist,
)


class LungDataModule(pl.LightningDataModule):
    '''
    This class used to setup the data for the training and validation.
    '''
    def __init__(self, 
        file_path:Union[str, os.PathLike],
        train_transform:Compose,
        val_transform:Compose,
        batch_size:int=1,
        num_workers:int=1) -> None:
        '''
        Initialize the class.

        :param file_path: The path of the data.
        :param type: The type of the data ['train', 'val', 'test'].
        :param batch_size: The batch size.
        :param transform: The transform function.
        '''
        super(LungDataModule, self).__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self, stage:Optional[str]=None):
        '''
        Setup the data.
        '''
        # prepare transforms standard to dataset
        trainlist = load_decathlon_datalist(self.file_path, True, 'train')
        vallist = load_decathlon_datalist(self.file_path, True, 'val')

        self.train_ds = CacheDataset(data=trainlist, transform=self.train_transform, cache_num=6, cache_rate=1.0, num_workers=5)
        self.val_ds = CacheDataset(data=vallist, transform=self.val_transform, cache_num=6, cache_rate=1.0, num_workers=5)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
