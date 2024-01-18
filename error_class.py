# general error class
from abc import ABC,abstractproperty

class Error(ABC):
    def __init__(self,predictions,observations):
        self._predictions = predictions
        self._observations = observations
    
    @property
    def predictions(self):
        return self._predictions
    
    @property
    def observations(self):
        return self._observations
    
    @abstractproperty
    def error_summary(self):
        pass
