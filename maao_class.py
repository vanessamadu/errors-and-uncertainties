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
        residuals_arr = np.zeros(len(self.predictions))
        for ii in range(len(residuals_arr)):
            if (self.predictions[ii] == np.zeros(2)).all() and (self.observations[ii]==np.zeros(2)).all():
                pred = np.ones(2)
                obs = np.ones(2)
            else:
                obs = self.observations[ii]
                pred = self.predictions[ii]
            residuals_arr[ii] = np.dot(pred,obs)/(linalg.norm(pred)*linalg.norm(obs))
        return residuals_arr
    
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
    