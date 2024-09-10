import torch

from .VariationalLayer import VariationalLayer
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
import numpy as np


class IsotropicGaussian(nn.Linear, VariationalLayer):
    '''
        This implementation is for an Isotropic Gaussian Variational Layer.
    '''
    def __init__(self, in_features, out_features) -> None:
        super(IsotropicGaussian, self).__init__(in_features, out_features, bias=True)
        self.log_sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma_bias = Parameter(torch.Tensor(out_features))
        
        torch.nn.init.xavier_normal_(self.log_sigma_weight)
        self.log_sigma_bias.data.fill_(0)
        
        self.mu, self.sigma = None, None

    def forward(self, x: torch.Tensor, n_samples=1) -> torch.Tensor:
        mu = F.linear(x, self.weight, self.bias) 
        sigma = torch.exp(0.5 * F.linear(x, self.log_sigma_weight, self.log_sigma_bias)) # log_sigma = 0.5 * log(sigma^2)

        self.mu, self.sigma = mu, sigma

        # Reparameterization trick
        # eps = torch.normal(0, torch.ones_like(sigma))
        # return mu + sigma * eps

        return self._reparametrize_n(mu, sigma, n_samples)

    def kl_reg(self) -> torch.Tensor:
        # KL-Divergence regularization
        # p ~ N(0, I)
        k = self.bias.size(0) # Dimension of Normal Distribution
        mu = self.mu.pow(2).sum(1) # mu^2
        det_sigma = self.sigma.prod(1) # determinant of sigma
        tr_sigma = self.sigma.sum(1) # trace of sigma

        kl = 0.5*(mu + tr_sigma - torch.log(det_sigma) - k)
        return kl.mean()

    def _reparametrize_n(self, mu, sigma, n=1):
        '''
            Reparameterization trick for n samples.
        '''
        def expand(v):
            assert torch.is_tensor(v)
            return v.expand(n, *v.size())
        
        mu = expand(mu)
        sigma = expand(sigma)
        eps = torch.normal(0, torch.ones_like(sigma))
        return (mu + sigma * eps)
    