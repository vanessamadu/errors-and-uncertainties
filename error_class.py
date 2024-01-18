# general error class
from abc import ABC,abstractproperty
class Error(ABC):
    
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