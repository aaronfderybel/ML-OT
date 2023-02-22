from pandas import DataFrame
from torch.utils.data import Dataset

class InterfaceModel():
    def __init__(self, data: Dataset):
        """ Initialize model and save dataset object in this class"""
        pass
    def fit(self) -> None:
        """Apply model specific fit function"""
        pass
    def predict(self) -> (DataFrame, DataFrame):
        """Return two pandas dataframes with predictions"""
        pass
    def evaluate(self, pred_train: DataFrame, pred_test: DataFrame) -> None:
        """Calculate standard evaluation scores for model and save in outputs"""
        pass
    
    def get_class_names(self) -> list:
        """Return class names in list"""
        pass