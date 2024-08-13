from abc import ABC, abstractmethod

class VariationalLayer(ABC):
    r'''
        Abstract base class for variational models which is mandatory to define
        the KL divergence.
    '''
    @abstractmethod
    def kl_reg(self):
        pass