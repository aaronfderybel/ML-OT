import hydra

@hydra.main(version_base=None, config_path="configs/", config_name="config_hpsearch.yaml")
def app(config):
    from src.workflow import WorkFlow
    
    dataset = hydra.utils.instantiate(config.dataset)
    model = hydra.utils.instantiate(config.model)
    wf = WorkFlow(dataset, model)
    wf.execute(config=config, save_objects=False)
    
    return float(model.score(wf.dataset))
    
if __name__ == "__main__":
    app()
