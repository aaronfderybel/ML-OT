import logging
import hydra
import numpy as np
import xgboost as xgb

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.interfaces import InterfaceData, InterfaceModel
from src.evaluation import cf_report

class RandomForest(InterfaceModel):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
        self.model_type = "Supervised"
        
    def fit(self, dataset: InterfaceData):
        self.model.fit(dataset.x_train, dataset.y_train.values.reshape(-1,))
        return None
    
    def predict(self, dataset: InterfaceData):
        return self.model.predict(dataset.x_train), self.model.predict(dataset.x_test)
    
    def evaluate(self, dataset: InterfaceData, preds):
        cf_report(self, dataset, preds)
    
    def score(self, dataset: InterfaceData):
        y_true = dataset.y_test.values.reshape(-1)
        preds = self.model.predict(dataset.x_test)
        return f1_score(y_true, preds, average='macro')
    
    def get_class_names(self, dataset: InterfaceData):
        return list(self.model.classes_)
        
        
class Xgb(InterfaceModel):
    def __init__(self, verbose: bool, **kwargs):
        self.model_type= "Supervised"
        self.kwargs = kwargs
        self.verbose = verbose
        self.label_enc = None
        self.model = None

    
    def fit(self, dataset: InterfaceData):
        y_train, y_test = dataset.y_train.values.ravel(), dataset.y_test.values.ravel()
        le = LabelEncoder().fit(y_train) #xgb expects class names to be 0 through n-1
        dataset.y_train = le.transform(y_train)
        dataset.y_test = le.transform(y_test)
        self.label_enc = le
        
        n_classes = len(np.unique(y_train))
               
        if  n_classes > 2:
            obj = 'multi:softmax'
            metric = 'mlogloss'
        else:
            obj = 'binary:hinge'
            metric = 'logloss'
            n_classes= None
        
        self.model = xgb.XGBClassifier(objective=obj, num_class=n_classes, 
                                       eval_metric=metric, **self.kwargs)
        
        
        evaluation = [( dataset.x_train, dataset.y_train.reshape(-1)), 
                      ( dataset.x_test, dataset.y_test.reshape(-1,))]
    
        self.model.fit(dataset.x_train, dataset.y_train.reshape(-1,),
                       eval_set=evaluation, verbose=self.verbose)
    
    
    def predict(self, dataset: InterfaceData):
        pred_train, pred_test = self.model.predict(dataset.x_train), self.model.predict(dataset.x_test)
        return pred_train, pred_test
    
    def evaluate(self, dataset: InterfaceData, preds):
        encoded_labels = list(np.unique(dataset.y_train))
        original_names = self.label_enc.inverse_transform(encoded_labels).astype('str')
        cf_report(self, dataset, preds, encoded_labels, original_names)
    
    def score(self, dataset: InterfaceData):
        y_true = dataset.y_test
        preds = self.model.predict(dataset.x_test)
        return f1_score(y_true, preds, average='macro')
    
    def get_class_names(self, dataset: InterfaceData):
        encoded_labels = list(np.unique(dataset.y_train))
        return self.label_enc.inverse_transform(encoded_labels).astype('str')
        