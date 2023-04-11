import pandas as pd

from src.interfaces import InterfaceData
from src.utils.nprobe import preprocess_nprobe, postprocess_nprobe

class NprobeData(InterfaceData):
    def load(self):
        self.data = pd.read_csv(self.data_path, delimiter=",")
        
    def preprocess(self):
        self.data = preprocess_nprobe(self.data)
        
    def postprocess(self):
        self.x_train, self.x_test = postprocess_nprobe(self.x_train, self.x_test)
            
        