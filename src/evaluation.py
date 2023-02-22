from numpy import array
import hydra

from typing import Optional
from torch.utils.data import Dataset
from sklearn.metrics import classification_report

from src.model_interface import InterfaceModel


def cf_report(dataset: Dataset, model: InterfaceModel,  pred_train: array , pred_test: array,
             class_values: Optional[list[int]] =None, target_values: Optional[list[str]] =None):
    
    cf_train = classification_report(y_true=dataset.y_train, y_pred=pred_train, 
                                     labels=class_values, target_names=target_values)
    
    cf_test = classification_report(y_true=dataset.y_test, y_pred=pred_test,
                                   labels=class_values, target_names=target_values)

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    with open(f"{output_dir}/{model.__class__.__name__}.txt",'w') as f:
        f.write('train-set\n')
        f.write('-------------------------\n')
        f.write(cf_train)
        f.write('-------------------------\n')
        f.write('test-set\n')
        f.write(cf_test)

    print('------------------------------\n')
    print('full classification report\n')
    print('------------------------------\n')
    print('train set')
    print(cf_train)
    print('test set')
    print(cf_test)
