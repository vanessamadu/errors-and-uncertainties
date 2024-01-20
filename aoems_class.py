# Angle Offset Error Metrics
from maao_class import MAAO
import numpy as np
import pandas as pd
from numpy import linalg

class AOEMS:
    def __init__(self,predictions,observations):
        self._predictions = np.array(predictions)
        self._observations = np.array(observations)
    
    # class variable: error tolerance
    tol = 10e-6

    @property
    def predictions(self):
        return self._predictions
    
    @property
    def observations(self):
        return self._observations
    
    @staticmethod

    def rotation_matrix(residual):
        sine_angle = np.sqrt(1-residual**2)
        return np.array([[residual, -sine_angle],[sine_angle,residual]])
    
    # angle offset error metrics

    @property
    def maao_all(self):
        return MAAO(self.predictions,self.observations)
    
    ## split up into anti-clockwise offset, clockwise offset, no offset

    @property
    def no_offset_indices(self):
        return np.array([ii for ii in range(len(self.maao_all.residuals)) if 1-self.maao_all.residuals[ii] < self.tol])
    
    @property
    def anti_clockwise_offset_indices(self):
        return np.array([ii for ii in range(len(self.maao_all.residuals)) if 
                         linalg.norm(self.maao_all.predictions[ii] - 
                                     np.matmul(
                                         __class__.rotation_matrix(
                                             self.maao_all.residuals[ii]),self.maao_all.observations[ii]
                                             ))<self.tol])