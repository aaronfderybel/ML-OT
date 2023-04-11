# Machine Learning in Operational Technology (ML-OT)
ML-OT is a toolbox to apply different algorithms and analyses on network data.
1. Different options for experiments can be set with the commandline and easily reproduced. The Open source project Hydra from facebook research provides this functionality.
2. Datasets made by different parsers for network data are supported f.e. tshark, nProbe.
3. Automatic evaluation and analyses o.a. statistics, explainable AI, ..

## Installation

Install pytorch with instructions from official website https://pytorch.org/get-started/locally/
This depends on specific operating system, hardware and software.
```console
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
```
Install remaining libraries of the requirements file.
```console
pip install -r requirements.txt
```
This command turns on tab completion for hydra.
```console
eval "$(python main.py -sc install=bash)"
```

When you make use of an virtual environment you can add this command to the activation script.
This will automatically activate tab completion when entering your virtual environment.
Use the absolute pad to the `main.py` script.
```console
echo 'eval "$(python <path_main_script> -sc install=bash)"' | tee -a env/bin/activate
```

## Usage

### Experiments with terminal
You can smoothly select different options for experiments and tab completion is enabled. 
This recording is a short showcase of some simple experiments.
[![asciicast](https://asciinema.org/a/Fa6K17UslsOcGqka3h8nmLak3.svg)](https://asciinema.org/a/Fa6K17UslsOcGqka3h8nmLak3)

Using this procedure you can explore the different configurations.
To apply one or more algorithms for one or more datasets add the `--multirun` arg to the command:
```console
python main.py --multirun dataset=tshark,nprobe model=randomforest,XGB
```
You can also do a **grid search** or sweep on a single algorithm:
```console
python main.py --multirun dataset=tshark model=randomforest model.n_estimators=2,5,10 model.max_depth=5,10,20
```
To perform a smarter hyperparameter search make use of `hp_search.py`:
```console
python hp_search.py --multirun dataset=tshark model=randomforest 'model.n_estimators=range(10,200)' 'model.max_depth=range(5,20)'
```
This will make use of the nevergrad plugin for hydra to search the parameter space and returns the parameter with the best average f1-score.
You can find more information on how to use nevergrad [here](https://hydra.cc/docs/plugins/nevergrad_sweeper/#defining-the-parameters)

### Experiments through config
Hydra uses a collection of config files in the background for the tab completion in terminal.
The config files can be adapted directly to have more extended control over the experiment.

The `configs` directory contains three important subdirectories:
1. `dataset`, contains the different options for datasets such as test size, data directory and whether labels should be binary or multi-class.
2. `model`, contains the different hyperparameters.
3. `analysis`, contains the different functions to apply after training for each model.

> Note that there are two types of models: classification and anomaly. Anomaly can only handle binary labels and classification can handle both binary and multi-class labels. 

To adapt the parameter search space used by the hp_search.py script adapt the config file `configs/hydra/sweeper/hyperparam_search.yaml`.
The default options here are for a randomforest model with the number of trees between 10 and 150. 
```yaml
parametrization:
    model : randomforest
    model.n_estimators:
        lower: 10
        upper: 150
        integer: True
```

You can also create a new config file, f.e. `my_search_space`, under the same directory and adapt the top level config file `configs/config_hpsearch.yaml` to:
```yaml
defaults:
    - _self_
    - dataset: tshark
    - model: randomforest
    - override hydra/sweeper: my_search_space  
```

This way you can maintain different experiments easily.

### Outputs

In the `outputs` directory you can find results of experiments with a single option of configs.
* main.log file containing all the executed steps.
* An evaluation for the model, report on the scores for the different classes.
* Analysis folder with different produced graphs.

These graphs are produced by different functions and they can be changed from the config files in directory `configs/analysis` each of the models contain different config files.

In the `multirun` directory you can find results of experiments with multiple options for config or sweeps. For grid search sweeps it will contain the same output as a normal experiment but in different subdirectories.For sweeps with nevergrad it will contain the different scores for each hyperparameter setting and the best found hyperparameters.


## Extendability
Datasets, models and analysis functions can be added to this project.

### Dataset
Dataset classes extend an interface.

```python
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
```

#### Step 1
To add a new dataset, extend the InterfaceData class in a python file under `src/datasets/`

#### Step 2
Write seperate functions for loading, preprocessing and postprocessing. You could also adapt the constructor, most of the time this will not be necessary.

#### Step 3
Add a new yaml file in the directory `configs/dataset`.\
It should target the new dataset class you created, a data path, test size, label_type and shuffle argument.
The tshark config looks like this:
```yaml
_target_: src.datasets.tshark.TsharkData
data_path: 'Data/tshark/scenario1'
label_type: 'binary'
test_size: 0.3
shuffle: False
```

### Model

Model classes extend an interface. 
```python
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
```

To add a new model 
#### Step 1
Extend the InterfaceModel class in a python file under `src/models/`. 

#### Step 2
Implement the different functions and the constructor of your subclass. 

#### Step 3
Add a new yaml file in the directory `configs/model`. 
It should target the new model class you created and add the hyperparameters of the model.

The randomforest config looks like this:
```yaml
_target_: src.models.classification.RandomForest #name of class
verbose: 0
n_estimators: 10
criterion: "gini"
max_depth: null #null converted to None by Hydra
max_features: "sqrt"
max_leaf_nodes: null
n_jobs: null
class_weight: null
```

### Analysis functions
Analysis functions follow the same interface
```python
def new_analysis_func(dataset: InterfaceData, model: InterfaceModel, **kwargs):
    ...
```
Each analysis function should accept a dataset object, model object and any additional arguments that are needed.
To add a new function

#### Step 1
Define a new function using the interface described above. `src.explain` contains a number of analysis functions.

#### Step 2
Add the analysis function to the config of a model under `configs/analysis/<model_name>.yaml`.\
It should target your newly made analysis function with *\_target\_*.\
It should also contain a data and model argument with *???*. This means it will get filled in by your chosen dataset and model objects in the experiment. If there are any additional arguments expected by the function you can add them under model.

For example randomforest contains a function to explain feature importance with shap library.\
shaptotal can be any chosen name and will be displayed in the logs.
```yaml
shaptotal:
    _target_: src.explain.shap_importance
    dataset: ???
    model: ???
```


## Relevant Documentation

* Hydra
    - nevergrad: [link](https://hydra.cc/docs/plugins/nevergrad_sweeper/)
    - instantiate objects/call functions with hydra: [link](https://hydra.cc/docs/1.1/advanced/instantiate_objects/overview/)
* Scikit-Learn [link](https://scikit-learn.org/stable/)
* Python Outlier Detection (PyOD) [link](https://github.com/yzhao062/pyod)
* Extreme Gradient Boosting (XGB) [link](https://github.com/dmlc/xgboost/)
## References

**Datasets**
* nprobe UNSW'15 https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA4
* tshark lemay https://github.com/antoine-lemay/Modbus_dataset
* Proprietary Datasets generated for the GAICIA project, more information [link](https://gaicia.ic4.be/)

**Papers**
* [link](https://www.usenix.org/conference/cset16/workshop-program/presentation/lemay) Providing SCADA Network Data Sets for Intrusion Detection Research. (A. Lemay & J. M. Fernandez)
* [link](https://arxiv.org/abs/1905.11757) Evaluation of machine learning based anomaly detection algorithms on modbus/TCP data set. (SD.  Anton & S. Kanoor)
* [link](https://arxiv.org/abs/2011.09144) NetFlow Datasets for Machine Learning-based Network Intrusion Detection Systems. (M. Sarhan & S. Layeghy)