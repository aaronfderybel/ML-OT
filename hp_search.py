import hydra
import logging

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs/", config_name="config_hpsearch.yaml")
def app(config) -> float:
    #lazy imports for faster tab completion
    import pickle
    from sklearn.metrics import f1_score
    import os
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    
    name_dataset = config.dataset._target_.split('.')[-1]
    data_dir = f"{output_dir}/../cache"
    path_file = f"{data_dir}/{name_dataset}.pkl"
    #RAUWE DATASET INLADEN
    if not os.path.exists(path_file):
        log.info(f"Instantiating dataset object of type {name_dataset}")
        dataset = hydra.utils.instantiate(config.dataset)
        #PREPARING DATASETS = PREPROCESSING+SPLITTING+POSTPROCESS
        log.info(f"preparing dataset using test size {config.dataset.test_size}")
        dataset.prepare()
        
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        
        with open(path_file, 'wb') as f:
            pickle.dump(dataset, f)
    else: 
        with open(path_file, 'rb') as f:
            dataset = pickle.load(f)
    
    
    
    #MODEL OPZETTEN EN TRAINEN
    log.info(f"Training Model of type {config.model._target_.split('.')[-1]}")
    log.info(f"With parameters {config.model}")
    model = hydra.utils.instantiate(config.model, data=dataset)
    model.fit()
    
    #PREDICTIES MAKEN EN EVALUEREN
    pred_train, pred_test = model.predict()
    model.evaluate(pred_train, pred_test)
    
    return float(f1_score(dataset.y_test, pred_test, average='macro'))
    
        
if __name__ == "__main__":
    app()
    
