# Speed Error Metrics Class
from mae_class import MAE
import numpy as np

class SEMs:
    def __init__(self,predictions,observations):
        self._predictions = np.array(predictions)
        self._observations = np.array(observations)

    @property
    def predictions(self):
        return self._predictions
    
    @property
    def observations(self):
        return self._observations
    
    ## split up into over estimated speed, under estimated speed, and correct speed

    @property
    def over_estimate_speed_indices(self):
        return [i for i in range(len(self.residuals)) if self.residuals[i] > 0]

    @property
    def under_estimate_speed_indices(self):
        return [i for i in range(len(self.residuals)) if self.residuals[i] < 0]
    
    @property
    def correct_estimate_speed_indices(self):
        return [i for i in range(len(self.residuals)) if self.residuals[i] == 0]
    
    # speed error metrics

    @property
    def mae_speed(self):
        return MAE(self.predictions,self.observations)
    


    
    