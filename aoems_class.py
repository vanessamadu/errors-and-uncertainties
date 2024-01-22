# Angle Offset Error Metrics
from maao_class import MAAO
import numpy as np
import pandas as pd
from numpy import linalg

class AOEMs:
    def __init__(self,predictions,observations):
        self._predictions = np.array(predictions)
        self._observations = np.array(observations)

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
        return vector/linalg.norm(vector)
    
    # angle offset error metrics

    @property
    def maao_all(self):
        return MAAO(self.predictions,self.observations)
    
    ## split up into anti-clockwise offset, clockwise offset, no offset

    @property
    def no_offset_indices(self):
        if len(self.maao_all.residuals[self.maao_all.defined_residual_indices]) == 0:
            return np.array([])
        return np.nonzero(np.isclose(1,self.maao_all.residuals[self.maao_all.defined_residual_indices]))[0]
    
    @property
    def anti_clockwise_offset_indices(self):
        remaining_indices = np.setdiff1d(range(len(self.predictions)),self.no_offset_indices)
        return np.array([ii for ii in remaining_indices if 
                         (np.allclose(
                             __class__.unit_vector(self.maao_all.predictions[ii]), 
                                     np.matmul(__class__.anti_clockwise_rotation_matrix(self.maao_all.residuals[ii]),__class__.unit_vector(self.maao_all.observations[ii]).T))
                                            )],dtype=int)
    
    @property
    def clockwise_offset_indices(self):
        return np.setdiff1d(range(len(self.predictions)),np.concatenate((self.no_offset_indices,self.anti_clockwise_offset_indices)))
    
    @property
    def clockwise_anticlockwise_no_undefined_proportions(self):
        return np.array([len(part)/len(self.predictions) for part in 
                [self.anti_clockwise_offset_indices,
                    self.clockwise_offset_indices,
                    self.no_offset_indices,
                    np.ones(len(self.predictions)-len(self.maao_all.defined_residual_indices))]])
    
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
            if (err_metrics[ii].error) == 'undefined':
                err = 'undefined'
            else:
                err = np.rad2deg(err_metrics[ii].error)
            if err_metrics[ii].uncertainty == 'undefined':
                uncert = 'undefined'
            else:
                uncert = np.rad2deg(err_metrics[ii].uncertainty)

            err_summary[f"{err_metric_names[ii]} Error: {err_metrics[ii].error_type}"]=f"{err} degrees"
            err_summary[f"{err_metric_names[ii]} Uncertainty: {err_metrics[ii].uncertainty_type}"] = f"{uncert} degrees"
        err_summary["Proportion of Anticlockwise/Clockwise/No/Undefined Angle Offset"]=f"{self.clockwise_anticlockwise_no_undefined_proportions*100}%"
        return pd.Series(err_summary)