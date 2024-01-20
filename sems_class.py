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
    
    # speed error metrics
    @property
    def mae_speed(self):
        return MAE(self.predictions,self.observations)
    
    ## split up into over estimated speed, under estimated speed, and correct speed

    @property
    def over_estimate_speed_indices(self):
        return np.array([i for i in range(len(self.mae_speed.residuals)) if self.mae_speed.residuals[i] > 0])

    @property
    def under_estimate_speed_indices(self):
        return np.array([i for i in range(len(self.mae_speed.residuals)) if self.mae_speed.residuals[i] < 0])
    
    @property
    def correct_estimate_speed_indices(self):
        return np.array([i for i in range(len(self.mae_speed.residuals)) if self.mae_speed.residuals[i] == 0])

    @property
    def over_under_correct_proportions(self):
        return np.array([len(part)/len(self.predictions) for part in 
                [self.over_estimate_speed_indices,
                    self.under_estimate_speed_indices,
                    self.correct_estimate_speed_indices]])

    @property
    def ma_overestimated_e(self):
        return MAE(self.predictions[self.over_estimate_speed_indices],
                   self.observations[self.over_estimate_speed_indices])
    
    @property
    def ma_underestimated_e(self):
        return MAE(self.predictions[self.under_estimate_speed_indices],
                   self.observations[self.under_estimate_speed_indices])

    ## Error-inherited property overwrites
    @property
    def error_summary(self):
        err_metrics = [self.mae_speed,self.ma_overestimated_e,self.ma_underestimated_e]
        err_metric_names = ["MAE over all Speeds","MAE for Overestimated Speeds", "MAE for Underestimated Speeds"]
        err_summary = {}
        for ii in range(len(err_metrics)):
            err_summary[f"{err_metric_names[ii]} Error: {err_metrics[ii].error_type}":err_metrics[ii].error]
            err_summary[f"{err_metric_names[ii]} Uncertainty: {err_metrics[ii].uncertainty_type}"
                        :err_metrics[ii].uncertainty]
        err_summary["Proportion of Over/Under/Correct Estimates of Speed":f"{self.over_under_correct_proportions*100}%"]
        return err_summary


    
    