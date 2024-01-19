from error_class import Error
import numpy as np
from numpy import linalg

class MAAO(Error):
    def __init__(self,predictions,observations):
        self._predictions = np.array(predictions)
        self._observations = np.array(observations)

    @property
    def predictions(self):
        return self._predictions
    
    @property
    def observations(self):
        return self._observations
    
    @property
    def residuals(self):
        'cosine of the angle between each prediction and observation'
        return np.array([np.dot(pred,obs)/(linalg.norm(pred)*linalg.norm(obs)) 
                            for pred,obs in zip(self.predictions,self.observations)])
    
    @staticmethod
    def maao(residuals_arr):
        return np.mean(np.arccos(residuals_arr))
    
    ## Error-inherited properties
    @property
    def error_type(self):
        return "maao"
    
    @property
    def error(self):
        return __class__.maao(self.residuals)