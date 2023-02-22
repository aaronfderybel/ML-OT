import hydra

@hydra.main(version_base=None, config_path="configs/", config_name="config.yaml")
def app(config):
    #lazy imports for faster tab completion
    import pickle
    import os
    
    from src.visuals import loghist 
    from src.explain import ECOD_outliers
    from hydra.utils import get_original_cwd
    
    import logging
    log = logging.getLogger(__name__)

    
    #maak analysis subdirectory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    os.mkdir(f"{output_dir}/analysis")
    
    #write down output_dir for overzichtsgrafiek
    if not os.path.exists(f"{get_original_cwd()}/cache"):
        os.mkdir(f"{get_original_cwd()}/cache")
        
    with open(f"{get_original_cwd()}/cache/.dir_latest_run", "w") as f:
        f.write(f"{output_dir[:output_dir.rindex('/')]}")
    
    #RAUWE DATASET INLADEN
    name_dataset = config.dataset._target_.split('.')[-1]
    
    data_dir = f"{output_dir}/../cache"
    path_file = f"{data_dir}/{name_dataset}.pkl"
    
    # indien multirun en dataset al geprepareerd is laad deze in.
    # anders prepareer dataset en sla deze op in cache voor multi-run
    if "multirun" in output_dir and os.path.exists(path_file):
        with open(path_file, 'rb') as f:
                dataset = pickle.load(f)
    else:
        log.info(f"Instantiating dataset object of type {name_dataset}")
        log.info(f"With data path: {config.dataset.data_path}")
        log.info(f"and classification_type: {config.dataset.classification_type}")
        dataset = hydra.utils.instantiate(config.dataset)
        #PREPARING DATASETS = PREPROCESSING+SPLITTING+POSTPROCESS
        log.info(f"preparing dataset using test size {config.dataset.test_size}")
        dataset.prepare()
        
        if "multirun" in output_dir:
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            f = open(path_file, 'wb')
            pickle.dump(dataset, f)
            f.close()
            
    
    #MODEL OPZETTEN EN TRAINEN
    log.info(f"Training Model {config.model._target_.split('models.')[1]}")
    model = hydra.utils.instantiate(config.model, data=dataset)
    model.fit()
    
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('data.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    #PREDICTIES MAKEN EN EVALUEREN
    pred_train, pred_test = model.predict()
    model.evaluate(pred_train, pred_test)
    
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
            hydra.utils.call(config.analysis[row], dataset=dataset, model=model)



if __name__ == "__main__":
    app()