import hydra
import numpy as np
import pandas as pd
import logging
import shap

from hydra.errors import InstantiationException
from xgboost import XGBClassifier, plot_importance
from typing import Union
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from pyod.utils.example import visualize

from src.models.classification import *
from src.models.anomaly import *

log = logging.getLogger(__name__)
     
def descriptive_stat_threshold(dataset: InterfaceData, model: Union[EcodModel, IsolationForest]):
    df = dataset.x_train.copy(deep=True)
    pred_score = model.model.decision_function(df)
    
    df['Anomaly_Score'] = pred_score
    df['Group'] = np.where(df['Anomaly_Score'] < model.model.threshold_, 'Normal', 'Outlier')

    # Now let's show the summary statistics:
    cnt = df.groupby('Group')['Anomaly_Score'].count().reset_index().rename(columns={'Anomaly_Score':'Count'})
    cnt['Count %'] = (cnt['Count'] / cnt['Count'].sum()) * 100 # The count and count %
    stat = df.groupby('Group').mean().round(2).reset_index() # The avg.
    stat = cnt.merge(stat, left_on='Group',right_on='Group') # Put the count and the avg. together
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    stat.to_csv(f'{output_dir}/analysis/outlier_stats_{model.__class__.__name__}.csv')
    
    del df
    
    return None

def ECOD_outliers(dataset: InterfaceData, model: EcodModel, query_string: str, n_plots: int):
    """
    Generate plots of the influence of individual features on total outlier score.

    Args:
        model: object of type ECODModel
        dataset: dummy variable for interface to work
        n_plots: number of plots to generate
        query_string: string passed to pandas.dataframe.query() function 
        can be used to filter dataset for specific rows. 
        Variables labels, pred_train can be used in the query using '@' symbol.
        examples: 
        '@labels==@pred_train' returns correctly predicted labels
        '@labels==@pred_train & @labels==1' returns correctly predicted attacks

    Returns:
        None
    """
    labels = dataset.y_train.values.reshape(-1,)
    ds = dataset.x_train.reset_index(drop=True).copy(deep=True)
    pred_train = model.predict(dataset)
    ds = ds.query(query_string)
    
    if len(ds)==0:
        log.warning(f"ECOD outlier analysis, no rows found with query string: {query_string}")
        return None
    
    cols = dataset.x_train.columns
    count = 0
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    for idx, _ in ds.iterrows():
        if count >= n_plots:
            return None
        
        plt.figure(figsize=(14,10))
        plt.xticks(rotation = 45)
        model.model.explain_outlier(idx, feature_names=cols)
        plt.savefig(f'{output_dir}/analysis/explain outlier row#{idx}.png')
        plt.close()
        count+=1
    
    del ds
    
    

def tree_importance(dataset: None, model: Union[RandomForestClassifier, IsolationForest]):
    """
    Generate built-in feature importance of Forest ensemble algorithmes using MDI.
    
    Args:
        model: object of type RandomForestClassifier or IsolationForest
        dataset: dummy variable for interface to work
    Returns:
        None
    """
    # If max_features is set to lower of total featues this analysis cannot be done
    # generate warning and skip this step
    feature_names = dataset.data.drop(columns=["label", "class"], errors="ignore").columns
    importances = model.model.feature_importances_
    
    if len(feature_names) != len(importances):
        log.warning(f"Leave default value of max_features to perform this analysis.\n number of features used: {len(importances)}, total number of features: {len(feature_names)}")
        return None
        
    std = np.std([tree.feature_importances_ for tree in model.model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI", fontsize=20)
    ax.set_ylabel("Mean decrease in impurity (MDI)", fontsize=18)
    fig.tight_layout()
    fig.set_size_inches(18.5, 10.5)
    plt.xticks(fontsize=20, rotation=90)
    plt.yticks(fontsize=20)
    plt.savefig(f'{output_dir}/analysis/{model.__class__.__name__}_importance.png')
    plt.close()
    return None
    
def xgb_importance(dataset: None, model: XGBClassifier):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    plt.figure(figsize=(14,10))
    plt.tight_layout(pad=0.5)
    plt.title("Feature Importance XGB", fontsize=20)
    xgb.plot_importance(model.model, xlabel="Importance Score", ax=plt.gca())
    plt.xticks(fontsize=20, rotation=90)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/analysis/{model.__class__.__name__}_importance.png')
    plt.close()
    return None

def shap_importance(dataset: InterfaceData, model: InterfaceModel):
    if model.__class__.__name__ in ["EcodModel"]:
        log.warning(f"{model.__class__.__name__} is not supported by shap, skipping analysis step")
        return None
    
    #use stratify sampling with max samples 1000 to reduce compute time
    sample = resample(dataset.x_train, n_samples=1000, replace=False, stratify=dataset.y_train)
    #some models are not supported directly by shap, see: https://github.com/slundberg/shap/issues/2399
    # print(model.__class__.__name__)
    # if model.__class__.__name__ in ["EcodModel"]:
    #     explainer = shap.Explainer(model.model.predict, sample)
    # else:
    #     explainer = shap.Explainer(model.model)
    
    explainer = shap.Explainer(model.model)
    values = explainer.shap_values(sample)
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    
    #Mean feature importance plot van alle klasses
    plt.figure()
    shap.summary_plot(values, sample, show=False, class_names=model.get_class_names(dataset), plot_type="bar")
    plt.savefig(f'{output_dir}/analysis/SHAP_{model.__class__.__name__}_meanimportance.png')
    plt.close()
    return None

def shap_beeswarm(dataset: InterfaceData, model: InterfaceModel):
    #some models are not supported by shap
    if model.__class__.__name__ in ["EcodModel"]:
        log.warning(f"{model.__class__.__name__} is not supported by shap, skipping analysis step")
        return None
    
    #use stratify sampling with max samples 1000 to reduce compute time
    sample = resample(dataset.x_train, n_samples=1000, replace=False, stratify=dataset.y_train)
    explainer = shap.Explainer(model.model)
    
    values = explainer.shap_values(sample)
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    
    if len(np.array(values).shape)==2:
        #for XGB models
        plt.figure()
        shap.summary_plot(values, sample, show=False)
        plt.savefig(f'{output_dir}/analysis/SHAP_{model.__class__.__name__}_importance_binary_class.png')
        plt.close()
        return None
    elif len(explainer.expected_value) == 2:
        #for binary classification
        plt.figure()
        shap.summary_plot(values[1], sample, show=False)
        plt.savefig(f'{output_dir}/analysis/SHAP_{model.__class__.__name__}_importance_binary_class.png')
        plt.close()
        return None
    else:
        #for multi-class classification
        for idx, class_name in enumerate(model.get_class_names(dataset)):
            plt.figure()
            shap.summary_plot(values[idx], sample, show=False)
            plt.savefig(f'{output_dir}/analysis/SHAP_{model.__class__.__name__}_importance_{class_name}.png')
            plt.close()
        
        return None
    
    


    