# general error class
from abc import ABC,abstractproperty

class Error(ABC):
    def __init__(self,error:float,uncertainty:float,error_type:str,uncertainty_type:str):
        self._error_type = error_type
        self._error = error
        self._uncertainty = uncertainty
        self._uncertainty_type = uncertainty_type

    @property
    def error(self):
        return self._error
    
    @property
    def uncertainty(self):
        return self._uncertainty
    
    @property
    def error_type(self):
        return self._error_type
    
    @property
    def uncertainty_type(self):
        return self._uncertainty_type
    
    @property
    def error_summary(self):
        return {self.error_type:self.error, self.uncertainty_type:self.uncertainty}