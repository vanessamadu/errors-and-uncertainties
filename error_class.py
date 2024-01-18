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
    
    @abstractproperty
    def uncertainty(self):
        pass
    
    @abstractproperty
    def error_type(self):
        pass
    
    @abstractproperty
    def uncertainty_type(self):
        pass
    
    @property
    def error_summary(self):
        return {self.error_type:self.error,
                self.uncertainty_type:self.uncertainty}

    @property
    def residuals(self):
        return np.array([pred-obs for pred,obs in zip(
            self.predictions,
            self.observations)])