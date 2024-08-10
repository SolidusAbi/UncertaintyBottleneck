from torch import optim

class EMAOptimizer(optim.Optimizer):
    ''' 
        Optimizer that computes a moving average of the variables.

        Based on [1], the exponential weighted moving average of the paramters
        during the trainig run will be used at test time.

        Example of usage for training:
        ```python
            optimizer = EMAOptimizer(model, Adam(model.parameters(), lr=1e-3), decay=.99)
            # ...
            optimizer.step()    
        ```

        Example of usage for evaluation:
        ```python
            optimizer.swap_weights()
            # Testing the model
            optimizer.swap_weights()
        ```

        References:
        -----------
        [1] Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K. (2016). 
            Deep variational information bottleneck. 
            arXiv preprint arXiv:1612.00410.
    '''
    def __init__(self, model, base_optimizer: optim.Optimizer, decay:float = .99):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        
        super(EMAOptimizer, self).__init__(base_optimizer.param_groups, base_optimizer.defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}
        self._state_dict = None
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.base_optimizer.step()
        self.__ema()

    def swap_weights(self) -> None:
        '''
            Swap the average and moving weights.

            This is a convenience method to allow one to evaluate the averaged weights
            at test time. Loads the weights stored in `self.shadow' into the model,
            keeping a copy of the original model weights. Swapping twice will return
            the original weights.
        '''
        if self._state_dict is None:
            self._state_dict = self.model.state_dict()
            self.model.load_state_dict(dict(self.shadow))
        else:
            self.model.load_state_dict(self._state_dict)
            self._state_dict = None
        
    def __ema(self) -> None:
        '''
            Update shadow variables by exponential moving average

            Only update the variables that require gradients?
        '''
        self.shadow = {
                name: param * self.decay + (1 - self.decay) * self.shadow[name] 
                for name, param in self.model.named_parameters() if param.requires_grad 
            }