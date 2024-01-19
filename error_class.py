# general error class
from abc import ABC,abstractproperty
import numpy as np

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
    def error_type(self):
        pass
    
    @abstractproperty
    def error(self):
        pass
    
    @property
    def uncertainty(self):
        return np.std(self.residuals)
    
    @property
    def uncertainty_type(self):
        return "std"
    
    @property
    def error_summary(self):
        return {self.error_type:self.error,
                self.uncertainty_type:self.uncertainty}

    @property
    def residuals(self):
        return np.array([pred-obs for pred,obs in zip(
            self.predictions,
            self.observations)])