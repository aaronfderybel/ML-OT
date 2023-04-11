import logging

from sklearn.model_selection import train_test_split

from src.interfaces import InterfaceData


log = logging.getLogger(__name__)

def supervised_split(dataset: InterfaceData):
    if dataset.label_type == 'binary':
        y = dataset.data[['label']]
    elif dataset.label_type =="multi":
        if 'class' not in dataset.data.columns:
            print("at log warning")
            log.warning("'class' column not present in dataset resorting to binary labels")
            y = dataset.data[['label']]
        else:
            y = dataset.data[['class']]

    x = dataset.data.drop(columns=['label','class'], errors='ignore')
    
    #if shuffle=True for dataset split by stratified sampling
    #otherwise chronological order is kept.
    if dataset.shuffle:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=dataset.test_size,\
                                                            shuffle=True, stratify=y)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=dataset.test_size,\
                                                            shuffle=False)
    return x_train, x_test, y_train, y_test

def unsupervised_split(dataset: InterfaceData):
    if dataset.label_type =="multi":
        log.warning("label_type 'multi' not supported for unsupervised learning, resorting to binary labels")
    if dataset.shuffle:
        log.warning("shuffle set to true will not change outcome in unsupervised learning.")
        
    x = dataset.data.drop(columns=['label','class'], errors='ignore')
    y = dataset.data[['label']]
    
    return x, y