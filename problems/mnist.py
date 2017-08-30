import os
# from problem import Problem
import torch
import torchvision as tv
from registry import register_problem

_data_root_path = 'data'

@register_problem
class MNIST:
    def __init__(self):
        # TODO postpone data preprocessing
        self._data_path = os.path.join(_data_root_path, self.__class__.__name__)
        self.train_pairs = tv.datasets.MNIST(root=self._data_path, train=True, download=True, transform=tv.transforms.ToTensor())
        self.val_pairs = tv.datasets.MNIST(root=self._data_path, train=False, download=True, transform=tv.transforms.ToTensor())
        self.test_pairs = self.val_pairs
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
        loader = torch.utils.data.DataLoader(self.train_pairs, batch_size=batch_size, shuffle=True)
        return loader

    def get_val_loader(self, batch_size):
        loader = torch.utils.data.DataLoader(self.val_pairs, batch_size=batch_size, shuffle=False)
        return loader

    def get_test_loader(self, batch_size):
        loader = torch.utils.data.DataLoader(self.test_pairs, batch_size=batch_size, shuffle=False)
        return loader
