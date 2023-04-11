from numpy import array
import hydra

from typing import Optional, Union
from torch.utils.data import Dataset
from sklearn.metrics import classification_report

from src.interfaces import InterfaceData, InterfaceModel

def cf_report(model: InterfaceModel, dataset: InterfaceData, preds: Union[tuple[array], array],
             class_values: Optional[list[int]] =None, target_values: Optional[list[str]] =None):
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    f = open(f"{output_dir}/{model.__class__.__name__}.txt",'w')
    
    if model.model_type in ["Supervised", "Semi-supervised"]:
        cf_train = classification_report(y_true=dataset.y_train, y_pred=preds[0], 
                                     labels=class_values, target_names=target_values)
    
        cf_test = classification_report(y_true=dataset.y_test, y_pred=preds[1],
                                   labels=class_values, target_names=target_values)
        
        f.write('train-set\n')
        f.write('-------------------------\n')
        f.write(cf_train)
        f.write('-------------------------\n')
        f.write('test-set\n')
        f.write(cf_test)
        f.close()
        print('------------------------------\n')
        print('full classification report\n')
        print('------------------------------\n')
        print('train set')
        print(cf_train)
        print('test set')
        print(cf_test)
        return None
        
    if model.model_type == "Unsupervised":
        cf = classification_report(y_true=dataset.y_train, y_pred=preds,\
                                   labels=class_values, target_names=target_values)
        f.write('results on complete dataset')
        f.write(cf)
        
        print('results on complete dataset')
        print(cf)
        return None