import pandas as pd
import logging
import os

from sklearn.model_selection import train_test_split

from src.utils.tsharkpcap import preprocess_tshark
from src.interfaces import InterfaceData

class TsharkData(InterfaceData):
    def load(self):
        df = pd.read_csv(self.data_path+'/data.csv', delimiter=",")
        if os.path.exists(self.data_path+'/labels.csv'):
            df_label = pd.read_csv(self.data_path+'/labels.csv', delimiter=";")
            self.data = df.merge(df_label, on='frame.number', how='left')
            del df, df_label
        else:
            self.data = df
            del df
    
    def preprocess(self):
        self.data = preprocess_tshark(self.data)