# general error class
class Error:
    def __init__(self,predictions,observations):
        self._predictions = predictions
        self._observations = observations
    
    @property
    def predictions(self):
        return self._predictions
    
    @property
    def observations(self):
        return self._observations