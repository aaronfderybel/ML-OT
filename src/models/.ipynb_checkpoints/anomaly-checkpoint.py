import logging
import hydra
import numpy as np

from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from sklearn.metrics import f1_score

from src.interfaces import InterfaceData, InterfaceModel
from src.evaluation import cf_report


log = logging.getLogger(__name__)

class IsolationForest(InterfaceModel):
    def __init__(self, **kwargs):
        self.model_type="Unsupervised"
        self.model = IForest(**kwargs)
        
    def fit(self, dataset: InterfaceData):
        self.model.fit(dataset.x_train.values)
    
    def predict(self, dataset: InterfaceData):
        return self.model.predict(dataset.x_train.values)
    
    def evaluate(self, dataset: InterfaceData, preds):
        cf_report(self, dataset, preds)
    
    def score(self, dataset: InterfaceData):
        y_true = dataset.y_train.values.reshape(-1)
        preds = self.model.predict(dataset.x_train)
        return f1_score(y_true, preds, average='macro')
        
    def get_class_names(self, dataset: InterfaceData):
        return [0,1]

class EcodModel(InterfaceModel):
    def __init__(self, **kwargs):
        self.model = ECOD(**kwargs)
        self.model_type ="Unsupervised"
        
    
    def fit(self, dataset: InterfaceData):
        self.model.fit(dataset.x_train.values)
    
    def predict(self, dataset: InterfaceData):
        return self.model.predict(dataset.x_train)
    
    def evaluate(self, dataset: InterfaceData, preds):
        cf_report(self, dataset, preds)
    
    def score(self, dataset: InterfaceData):
        y_true = dataset.y_train.values.reshape(-1)
        preds = self.model.predict(dataset.x_train)
        return f1_score(y_true, preds, average='macro')
    
    def get_class_names(self):
        return [0,1]
