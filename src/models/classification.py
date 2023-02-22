import logging
import hydra
import numpy as np
import xgboost as xgb

from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestClassifier

from src.model_interface import InterfaceModel
from src.evaluation import cf_report

class RandomForest(InterfaceModel):
    def __init__(self, data: Dataset, **kwargs):
        self.dataset = data
        self.model = RandomForestClassifier(**kwargs)
        
    def fit(self):
        self.model.fit(self.dataset.x_train, self.dataset.y_train.values.reshape(-1,))
        return None
    
    def predict(self):
        return self.model.predict(self.dataset.x_train), self.model.predict(self.dataset.x_test)
    
    def evaluate(self, pred_train, pred_test):
        cf_report(self.dataset, self.model, pred_train, pred_test)
    
    def get_class_names(self):
        return list(self.model.classes_)
        
        
class Xgb(InterfaceModel):
    def __init__(self, data: Dataset, verbose: bool, **kwargs):
        from sklearn.preprocessing import LabelEncoder
        
        y_train, y_test = data.y_train.values.ravel(), data.y_test.values.ravel()
        
        le = LabelEncoder().fit(y_train) #xgb expects class names to be 0 through n-1
        data.y_train = le.transform(y_train)
        data.y_test = le.transform(y_test)
        self.label_enc = le
        self.dataset = data
        self.verbose = verbose
        
        n_classes = len(np.unique(y_train))
               
        if  n_classes > 2:
            obj = 'multi:softmax'
            metric = 'mlogloss'
        else:
            obj = 'binary:hinge'
            metric = 'logloss'
            n_classes= None
        
        self.model = xgb.XGBClassifier(objective=obj, num_class=n_classes, 
                                       eval_metric=metric, **kwargs)
    def fit(self):
        evaluation = [( self.dataset.x_train, self.dataset.y_train.reshape(-1)), 
                      ( self.dataset.x_test, self.dataset.y_test.reshape(-1,))]
    
        self.model.fit(self.dataset.x_train, self.dataset.y_train.reshape(-1,),
                       eval_set=evaluation, verbose=self.verbose)
    
    
    def predict(self):
        pred_train, pred_test = self.model.predict(self.dataset.x_train), self.model.predict(self.dataset.x_test)
        #to return original class names for reporting
        # return self.label_enc.inverse_transform(pred_train), self.label_enc.inverse_transform(pred_test)
        return pred_train, pred_test
    
    def evaluate(self, pred_train, pred_test):
        encoded_labels = list(np.unique(self.dataset.y_train))
        original_names = self.label_enc.inverse_transform(encoded_labels).astype('str')
        
        cf_report(self.dataset, self.model, 
                  pred_train, pred_test,
                  encoded_labels, original_names)
    
    def get_class_names(self):
        class_ind = np.unique(self.dataset.y_train)
        return list(self.label_enc.inverse_transform(class_ind))
        