# Angle Offset Error Metrics
from maao_class import MAAO
import numpy as np
import pandas as pd

class AOEMS:
    def __init__(self,predictions,observations):
        self._predictions = np.array(predictions)
        self._observations = np.array(observations)
    
    @property
    def predictions(self):
        return self._predictions
    
    @property
    def observations(self):
        return self._observations