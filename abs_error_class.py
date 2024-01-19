from error_class import Error
import numpy as np

class MAE(Error):
    def __init__(self,predictions,observations):
        self._predictions = predictions
        self._observations = observations
    
    @property
    def predictions(self):
        return self._predictions
    
    @property
    def observations(self):
        return self._observations
    
    @staticmethod
    def mae(residuals_arr):
        return np.mean(np.abs(residuals_arr))
    
    ## Error-inherited properties

    @property
    def error_type(self):
        return "mae"
    
    @property
    def error(self):
        return __class__.mae(self.residuals)
    
    @property
    def uncertainty_type(self):
        return "std"
    
    @property
    def uncertainty(self):
        return np.std(self.residuals)