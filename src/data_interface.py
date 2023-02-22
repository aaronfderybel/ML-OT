from torch.utils.data import Dataset

class InterfaceData(Dataset):
    def __init__(self, data_path: str, test_size: float, **kwargs):
        """Initialize dataset object, expect at least data path and test_size"""
        self.data=None
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.iloc[idx,:]           
    
    def _preprocess(self):
        """Apply a specified preprocessing function here"""
        pass
        
    def _split_data(self):
        """Strategy to split data into train and test set"""
        pass
        
    def _postprocess(self):
        """Any processing applied after splitting data, such as rescaling"""
        pass
    
    def prepare(self):
        self._preprocess()
        self._split_data()
        self._postprocess()