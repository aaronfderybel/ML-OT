import hydra
import logging
import pickle
import os
        
from hydra.utils import get_original_cwd

from src.interfaces import InterfaceData, InterfaceModel
from src.utils.split import supervised_split, unsupervised_split

# volgens het design pattern https://refactoring.guru/design-patterns/builder
# hier gebruik ik wel twee objecten met een interface ipv 1.
# de workflow klasse zou je als een director kunnen zien van twee builder klasses model en dataset
# Het meest logische is om alle model-specifieke eigenschappen in het model object te steken en idem voor data object.
# vervolgens kan de director klasse alle nodige communicatie tussen de twee verwezenlijken. Model bevat of het een supervised, semi-supervised of unsupervised algoritme gaat in model_type.
# initialisatie van model en dataset worden ook door de gebruiker gedaan. Als dit eenmaal is gebeurd zou de gebruiker de verdere logica niet moeten weten om een correct experiment uit te voeren.
class WorkFlow():
    def __init__(self, dataset: InterfaceData, model: InterfaceModel):    
        self.dataset = dataset
        self.model = model
    
    def execute(self, config, save_objects):
        """Performs all the steps to perform an experiment for a specific model and dataset object"""
        log = logging.getLogger(__name__)
        
        # print("ANALYSIS IS:", help(config))
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        output_dir = hydra_cfg['runtime']['output_dir']
        name_dataset = config.dataset._target_.split('.')[-1]
        name_model = config.model._target_.split('.')[-1]
        
        path_data=f"{output_dir}/../{name_dataset}_{self.model.model_type.lower()}.pkl"
        print("path_data is", path_data)
        print("current directory", os.getcwd())
        
        if not os.path.exists(f"{output_dir}/analysis"):
            os.mkdir(f"{output_dir}/analysis")
        
        if "multirun" in output_dir and os.path.exists(path_data):
            with open(path_data, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            log.info(f"Loading data of type {name_dataset}")
            log.info(f"With data path: {config.dataset.data_path}")
            log.info(f"label_type: {config.dataset.label_type}")
            if self.model.model_type=="Unsupervised":
                log.info(f"and test size set to 0.0 for unsupervised training")
            else:
                log.info(f"and test size: {config.dataset.test_size}")
                
            self.dataset.load()
            self.dataset.preprocess()
        
            if self.model.model_type=="Supervised":
                self.dataset.x_train, self.dataset.x_test,\
                self.dataset.y_train, self.dataset.y_test = supervised_split(self.dataset)
                
            elif self.model.model_type=="Unsupervised":
                self.dataset.x_train, self.dataset.y_train = unsupervised_split(self.dataset)
            
            self.dataset.postprocess()
            
            if "multirun" in output_dir and not os.path.exists(path_data):
                f = open(path_data, 'wb')
                pickle.dump(self.dataset, f)
                f.close()
        
            
        
        log.info(f"Training Model {config.model._target_.split('models.')[1]}")
        log.info(f"which is an {self.model.model_type} type model")
        
        self.model.fit(self.dataset)
        predicts = self.model.predict(self.dataset)
        self.model.evaluate(self.dataset, predicts)
        
        if save_objects:
            os.mkdir(f"{output_dir}/cache")
            f = open(f"{output_dir}/cache/{name_dataset}_{self.model.model_type.lowercase()}.pkl", 'wb')
            pickle.dump(self.dataset,f)
            f.close()
            
            f = open(f"{output_dir}/cache/{name_model}.pkl", 'wb')
            pickle.dump(self.model, f)
            f.close()
            
        if "analysis" in config:
            target_functions = []
            for row in config.analysis:
                function_name = config.analysis[row]._target_.split('src.')[1]
                if function_name not in target_functions:
                    target_functions.append(function_name)

            if len(target_functions) != 0:
                log.info("Make visual and explainability graphs with following unique functions:")
                for idx, function in enumerate(target_functions):
                    log.info(f"{idx+1}. {function}")

                for row in config.analysis:
                    hydra.utils.call(config.analysis[row], dataset=self.dataset, model=self.model)

            
            
        
    