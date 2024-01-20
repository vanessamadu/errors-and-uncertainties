# Angle Offset Error Metrics
from maao_class import MAAO
import numpy as np
import pandas as pd
from numpy import linalg

class AOEMs:
    def __init__(self,predictions,observations):
        self._predictions = np.array(predictions)
        self._observations = np.array(observations)
    
    # class variable: error tolerance
    tol = 10e-5

    @property
    def predictions(self):
        return self._predictions
    
    @property
    def observations(self):
        return self._observations
    
    @staticmethod

    def anti_clockwise_rotation_matrix(residual):
        sine_angle = np.sqrt(1-residual**2)
        return np.array([[residual, -sine_angle],[sine_angle,residual]])
    
    def clockwise_rotation_matrix(residual):
        sine_angle = np.sqrt(1-residual**2)
        return np.array([[residual, sine_angle],[-sine_angle,residual]])
    
    @staticmethod
    def unit_vector(vector):
        if linalg.norm(vector) == 0:
            return vector
        return vector/linalg.norm(vector)
    
    # angle offset error metrics

    @property
    def maao_all(self):
        return MAAO(self.predictions,self.observations)
    
    ## split up into anti-clockwise offset, clockwise offset, no offset

    @property
    def no_offset_indices(self):
        return np.array([ii for ii in range(len(self.maao_all.residuals)) if np.isclose(1,self.maao_all.residuals[ii])])
    
    @property
    def anti_clockwise_offset_indices(self):
        return np.array([ii for ii in range(len(self.maao_all.residuals)) if 
                         (np.allclose(__class__.unit_vector(self.maao_all.predictions[ii]), 
                                     np.matmul(
                                         __class__.anti_clockwise_rotation_matrix(
                                             self.maao_all.residuals[ii]),__class__.unit_vector(self.maao_all.observations[ii]).T
                                             ))
                                             and not np.isclose(1,self.maao_all.residuals[ii]))],dtype=int)
    
    @property
    def clockwise_offset_indices(self):
        return np.array([ii for ii in range(len(self.maao_all.residuals)) if 
                         (np.allclose(__class__.unit_vector(self.maao_all.predictions[ii]), 
                                     np.matmul(
                                         __class__.clockwise_rotation_matrix(
                                             self.maao_all.residuals[ii]),__class__.unit_vector(self.maao_all.observations[ii]).T
                                             ))
                                             and not np.isclose(1,self.maao_all.residuals[ii]))],dtype=int)
    
    @property
    def clockwise_anticlockwise_no_proportions(self):
        return np.array([len(part)/len(self.predictions) for part in 
                [self.anti_clockwise_offset_indices,
                    self.clockwise_offset_indices,
                    self.no_offset_indices]])
    
    @property
    def maao_anticlockwise(self):
        return MAAO(self.predictions[self.anti_clockwise_offset_indices],
                    self.observations[self.anti_clockwise_offset_indices])
    
    @property
    def maao_clockwise(self):
        return MAAO(self.predictions[self.clockwise_offset_indices],
                    self.observations[self.clockwise_offset_indices])
    
    ## Error-inherited property overwrites
    @property
    def error_summary(self):
        err_metrics = [self.maao_all,self.maao_anticlockwise,self.maao_clockwise]
        err_metric_names = ["MAAO over all Angle Offsets","MAAO for Anticlockwise Offsets", "MAAO for Clockwise Offsets"]
        err_summary = {}
        for ii in range(len(err_metrics)):
            err_summary[f"{err_metric_names[ii]} Error: {err_metrics[ii].error_type}"]=f"{np.rad2deg(err_metrics[ii].error)} degrees"
            err_summary[f"{err_metric_names[ii]} Uncertainty: {err_metrics[ii].uncertainty_type}"] = f"{np.rad2deg(err_metrics[ii].uncertainty)} degrees"
        err_summary["Proportion of Anticlockwise/Clockwise/No Angle Offset"]=f"{self.clockwise_anticlockwise_no_proportions*100}%"
        return pd.Series(err_summary)