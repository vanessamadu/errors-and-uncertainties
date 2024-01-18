# general error class
from abc import ABC,abstractproperty
import numpy as np

class Error(ABC):
    
    @abstractproperty
    def predictions(self):
        pass

    @abstractproperty
    def observations(self):
        pass

    @abstractproperty
    def error(self):
        pass
    
    @property
    def uncertainty(self):
        pass
    
    @property
    def error_type(self):
        pass
    
    @property
    def uncertainty_type(self):
        pass
    
    @property
    def error_summary(self):
        pass

    @property
    def residuals(self):
        return np.array([pred-obs for pred,obs in zip(
            self.predictions,
            self.observations)])