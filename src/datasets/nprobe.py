import pandas as pd
import logging

from sklearn.model_selection import train_test_split

from src.data_interface import InterfaceData
from src.utils.nprobe import preprocess_nprobe, postprocess_nprobe

log = logging.getLogger(__name__)

class NprobeData(InterfaceData):
    def __init__(self, data_path: str, test_size: float, classification_type: str):
        self.data = pd.read_csv(data_path, delimiter=",")
        
        self.classification_type = classification_type
        self.test_size = test_size
        self.x_train, self.x_test = None, None
        self.y_train, self.y_test = None, None
    
    def _preprocess(self):
        self.data = preprocess_nprobe(self.data)
        
    def _split_data(self):
        if self.classification_type not in ['binary','multi']:
            log.warning(f"{self.classification_type} is an invalid value for classification_type,\
            should be 'binary or 'multi'")
            return None
                        
        elif self.classification_type == 'binary':
            if 'label' not in self.data.columns:
                log.warning(f"expected column with name 'label' to be in dataset for binary classification")
                return None
            
            y = self.data[['label']]
        elif self.classification_type == "multi":
            if 'class' not in self.data.columns:
                log.warning(f"expected column with name 'class' to be in dataset for multi-class classification")
                return None
            
            y = self.data[['class']]
            
        x = self.data.drop(columns=['label', 'class'], errors='ignore')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, stratify=y)
        self.x_train, self.x_test = x_train, x_test
        self.y_train, self.y_test = y_train, y_test
        
    def _postprocess(self):
        self.x_train, self.x_test = postprocess_nprobe(self.x_train, self.x_test)
        