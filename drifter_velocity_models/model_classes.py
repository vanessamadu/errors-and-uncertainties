'description: hierarchical model framework for the construction of my ocean drifter velocity model'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%% SET UP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
##### import packages #####
import numpy as np
from numpy import linalg
import pandas as pd
import math
from typing import List


##### load data #####
data = pd.read_hdf("ocean_data.h5")

class Model:
    '''
    this class will be the parent class of all ocean models that we will be using
    and it will define attributes and methods common to all specific model classes.
    '''
    def __init__(self,training_data:List[float],test_data:List[float]):
        ## model specifiers
        self.model_type = None
        ## data attributes
        self.training_data = training_data # data as pd dataframe
        self.test_data = test_data # data as pd dataframe
        ## for probabilistic regression models
        self.trained_distribution = None
        self.test_distribution = None

    #++++++++++++++++++++++++ STATIC METHODS ++++++++++++++++++++++++++++#
    @staticmethod
    def to_degrees(arr:List[float]):
        return np.rad2deg(arr)
    
    @staticmethod
    def to_cm_per_second(arr:List[float]):
        return np.multiply(100,arr)
    
    # -------------------- validation -------------------- #

    @staticmethod
    def check_coordinates(lon:float,lat:float):
        limits = {"lat":90.,"lon":180.}
        values = {"lat":lat,"lon":lon}

        for coord in values.keys():
            try:
                float(values[coord])
            except:
                raise ValueError(f"{coord} must be a real number")
            finally:
                if np.abs(values[coord])>limits[coord]:
                    raise ValueError(f"{coord} must be between -{limits[coord]} and {limits[coord]}")

    #++++++++++++++++++++++ MODEL PROPERTIES AND SETTERS +++++++++++++++++++++#
    # -------------------- properties ---------------------#
    @property
    def training_data(self):
        '(setter) data used to train'
        return self._training_data
    
    @property
    def test_data(self):
        '(setter) data used to test'
        return self._test_data
    
    # --------------------- setters ------------------------ #
    @training_data.setter
    def training_data(self,data_subset):
        'changes the value of the training_data property'
        self._training_data = data_subset

    @test_data.setter
    def test_data(self,data_subset):
        'changes the value of the test_data property'
        self._test_data = data_subset

