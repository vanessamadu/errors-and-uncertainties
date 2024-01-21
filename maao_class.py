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
            if (self.predictions[ii] == 0).all() or (self.observations[ii]==0).all():
                residuals_arr[ii] = 'undefined'
            else:
                obs = self.observations[ii]
                pred = self.predictions[ii]
                residuals_arr[ii] = np.dot(pred,obs)/(linalg.norm(pred)*linalg.norm(obs))
        return residuals_arr
    
    @property
    def defined_residual_indices(self):
        return np.array([ii for ii in range(len(self.residuals)) 
                         if self.residuals[ii] != 'undefined'])
    
    @staticmethod
    def maao(defined_residuals):
        # mean absolute angle offset over all residuals that are defined
        if len(defined_residuals) == 0:
            return 'undefined'
        return np.mean(np.arccos(defined_residuals))

    ## Error-inherited properties
    @property
    def error_type(self):
        return "maao"
    
    @property
    def error(self):
        return __class__.maao(self.residuals[self.defined_residual_indices])
    
    