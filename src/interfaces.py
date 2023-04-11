from pandas import DataFrame
from torch.utils.data import Dataset
from typing import Union

class InterfaceData(Dataset):
    def __init__(self, data_path: str, test_size: float, label_type: str, shuffle: bool):
        """Initialize dataset object with properties"""
        self.data_path = data_path
        self.test_size = test_size
        self.label_type = label_type
        self.shuffle = shuffle
        
        self.data=None
        self.x_train, self.x_test = None, None
        self.y_train, self.y_test = None, None
    
    def __len__(self):
        """method to retrieve length of data contained"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """method to retrieve row of data"""
        return self.data.iloc[idx,:]           
    
    def load(self) -> None:
        """Load in dataset and store in object"""
        pass
    
    def preprocess(self) -> None:
        """Apply a specified preprocessing function and store in object"""
        pass
        
    def postprocess(self) -> None:
        """Any processing applied after splitting data, such as rescaling store in object"""
        pass


class InterfaceModel():
    def __init__(self, **kwargs):
        """ Initialize model_type as 'supervised','semi-supervised' or 'unsupervised'
        and pass kwargs as hyperparameters model"""
        self.model_type=None
        pass
    
    def fit(self, dataset: InterfaceData) -> None:
        """Apply model specific fit function"""
        pass
    def predict(self, dataset: InterfaceData) -> Union[tuple[DataFrame, DataFrame], DataFrame]:
        """Return one or two pandas dataframes with predictions"""
        pass
    def evaluate(self, dataset: InterfaceData, preds) -> None:
        """Calculate standard evaluation scores for model and save in outputs"""
        pass
    
    def score(self, dataset: InterfaceData) -> float:
        """Get singular score on relevant piece of data, depends of model type"""
        pass
    
    def get_class_names(self, dataset: InterfaceData) -> list:
        """Return original class names in list or [0,1] for binary datasets"""
        pass