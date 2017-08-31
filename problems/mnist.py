import os
# from problem import Problem
import torch
import torchvision as tv
from registry import register_problem
from torch.utils.data.sampler import SubsetRandomSampler


_data_root_path = 'data'

@register_problem
class MNIST:
    def __init__(self):
        # TODO postpone data preprocessing
        self._data_path = os.path.join(_data_root_path, self.__class__.__name__)
        # Split the dataset into first 50K for train, last 10K for validation
        n_train, n_val = 50000, 10000
        self.train_indices = range(n_train)
        self.val_indices = range(n_train, n_train + n_val)
        self.train_pairs = tv.datasets.MNIST(root=self._data_path, train=True, download=True, transform=tv.transforms.ToTensor())
        self.val_pairs = tv.datasets.MNIST(root=self._data_path, train=True, download=True, transform=tv.transforms.ToTensor())
        self.test_pairs = tv.datasets.MNIST(root=self._data_path, train=False, download=True, transform=tv.transforms.ToTensor())
        super(MNIST, self).__init__()

    @property
    def specs(self):
        return {
            'input_shape': [1, 28, 28],
            'output_classes': 10,
            'train_size': len(self.train_pairs),
            'val_size': len(self.val_pairs),
            'test_size': len(self.test_pairs),
        }

    def get_train_loader(self, batch_size):
        loader = torch.utils.data.DataLoader(self.train_pairs, batch_size=batch_size, sampler=SubsetRandomSampler(self.train_indices), drop_last=True, num_workers=1)
        return loader

    def get_val_loader(self, batch_size):
        loader = torch.utils.data.DataLoader(self.val_pairs, batch_size=batch_size, sampler=SubsetRandomSampler(self.val_indices))
        return loader

    def get_test_loader(self, batch_size):
        loader = torch.utils.data.DataLoader(self.test_pairs, batch_size=batch_size, shuffle=False)
        return loader
