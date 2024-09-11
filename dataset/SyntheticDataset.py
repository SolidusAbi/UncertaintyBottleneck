import numpy as np
from abc import ABC, abstractmethod
from torch import tensor
from torch.utils.data import Dataset

class SyntheticDataset(Dataset, ABC):
    def __init__(self, n_samples:int, noise:float = .6) -> None:
        '''
            Parameters:
            -----------
                n_samples: int
                    Number of samples to generate
                
                noise: float
                    Noise magnitude
        '''
        super(SyntheticDataset, self).__init__()
        self.n_samples = n_samples
        self.noise = noise
        self.X, self.y = None, None

    @abstractmethod
    def generate_dataset(self) -> None:
        pass
    
class SyntheticDataset1(SyntheticDataset):
    '''
        Sinousoidal dataset with two frequencies. 

        La segunda variable determina la dirección y la magnitud de la incertidumbre
    ''' 
    def __init__(self, n_samples:int, noise:float=.6) -> None:
        super(SyntheticDataset1, self).__init__(n_samples, noise)
        self.generate_dataset()

    def generate_dataset(self) -> None:
        self.X = np.arange(0, 2*np.pi, 2*np.pi/self.n_samples)    

        # Determines direction and magnitude of Uncertainty
        noise_signal = np.arange(0, 5*np.pi, 5*np.pi/self.n_samples)
        epsilon = np.random.rand(self.n_samples)

        self.y = np.sin(2*self.X) + (np.sin(2*noise_signal + np.pi/4) * (self.noise*epsilon))
        
        # To Tensor
        self.X = self.__to_tensor(self.X).unsqueeze(1)
        self.X = (self.X - self.X.mean()) / self.X.std()
        self.y = self.__to_tensor(self.y)
        self.y = (self.y - self.y.mean()) / self.y.std()

    def __to_tensor(self, x):
        return tensor(x).float()
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx:int) -> tuple:       
        return self.X[idx], self.y[idx]
                                   