from torch import nn
from torch.nn.functional import cross_entropy
from ..nn import VariationalLayer

class SGVBL(nn.Module):
    r''' 
        Stochastic Gradient Variational Bayes (SGVB) Loss function.
        More details in https://arxiv.org/pdf/1506.02557.pdf and https://arxiv.org/pdf/1312.6114.pdf

        Parameters
        ----------
            model: model to train
            train_size: size of the training dataset
            mle: maximum likelihood estimation loss function
    '''

    def __init__(self, model, train_size, mle=cross_entropy):
        super(SGVBL, self).__init__()
        self.train_size = train_size
        self.net = model
        self.loss = mle

        self.variational_layers = []
        for module in model.modules():
            if isinstance(module, (VariationalLayer)):
                self.variational_layers.append(module)

    def forward(self, input, output, target, kl_weight=1.0):
        assert output.requires_grad
        kl = 0.0
        for layer in self.variational_layers:
            kl += layer.kl_reg(target)
            
        return self.train_size * (self.loss(input, output, reduction='mean') + kl_weight * kl)