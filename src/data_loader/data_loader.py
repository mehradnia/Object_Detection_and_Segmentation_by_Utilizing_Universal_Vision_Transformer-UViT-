import tensorflow as tf
from abc import ABC, abstractmethod
import os


class DataLoader:
    def __init__(self, data_dir: str, batch_size: int, image_size: tuple) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    @abstractmethod
    def _load_dataset(self, split):
        pass

    def set_datasets(self):
        self.train_ds = self._load_dataset('train')
        self.val_ds = self._load_dataset('val')
        self.test_ds = self._load_dataset('test')

        print('Data has been loaded successfully')

    def get_train_dataset(self):
        return self.train_ds

    def get_validation_dataset(self):
        return self.validation_ds

    def get_test_dataset(self):
        return self.test_ds
