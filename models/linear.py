from torch import nn
import numpy as np
from registry import register_model

@register_model
class LinearModel(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(LinearModel, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        flat_input_size, flat_output_size = np.prod(input_shape), np.prod(output_shape)
        self.linear = nn.Linear(int(flat_input_size), int(flat_output_size))

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        x = self.linear(x)
        x = x.view(bs, *self.output_shape)
        return x
