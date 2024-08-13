from torch import nn

from .VariationalLayer import VariationalLayer

class Sequential(nn.Sequential):
    def forward(self, x, *args):
        for module in self:
            if isinstance(module, VariationalLayer):
                x = module(x, *args) # Pass the additional arguments to the forward method
            else:
                x = module(x)
        return x