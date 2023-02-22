import logging
import hydra
import numpy as np

from torch.utils.data import Dataset
from pyod.models.ecod import ECOD
# from sklearn.ensemble import IsolationForest as Iforest
from pyod.models.iforest import IForest

from src.model_interface import InterfaceModel
from src.evaluation import cf_report


log = logging.getLogger(__name__)

class IsolationForest(InterfaceModel):
    def __init__(self, data: Dataset, verbose: int, **kwargs):
        #should be 2 unique label values for anomaly algorithms
        num_class = len(np.unique(data.y_train.values))
        assert num_class == 2, f"Isolation Forest expects number of unique label values to be 2, got {num_class}"
        
        self.dataset = data
        contamination = (data.y_train.values != 0).sum()/len(data.y_train)
        self.model = IForest(verbose=verbose, contamination=contamination, **kwargs)
        
    def fit(self):
        self.model.fit(self.dataset.x_train.values)
    
    def predict(self):
        return self.model.predict(self.dataset.x_train.values), self.model.predict(self.dataset.x_test.values)
    
    def evaluate(self, pred_train, pred_test):
        cf_report(self.dataset, self.model, pred_train, pred_test)
        
    def get_class_names(self):
        return [0,1]

class EcodModel(InterfaceModel):
    def __init__(self, data: Dataset):
        num_class = len(np.unique(data.y_train.values))
        assert num_class == 2, f"Ecod expects number of unique label values to be 2, got {num_class}"
            
        self.dataset = data
        contamination = (data.y_train.values != 0).sum()/len(data.y_train)
        self.model = ECOD(contamination=contamination)
        
    
    def fit(self):
        self.model.fit(self.dataset.x_train.values)
        log.info(f"ECOD params are: {str(self.model.get_params())}")
        return None
    
    def predict(self):
        return self.model.predict(self.dataset.x_train), self.model.predict(self.dataset.x_test)
    
    def evaluate(self, pred_train, pred_test):
        cf_report(self.dataset, self.model, pred_train, pred_test)
    
    def get_class_names(self):
        return [0,1]
