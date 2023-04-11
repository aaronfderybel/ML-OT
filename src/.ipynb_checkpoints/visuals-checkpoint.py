import hydra
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

from torch.utils.data import Dataset


def loghist(dataset: Dataset, model: None, alpha: float):
    y = dataset.data['label']
    df_normal = dataset.data[y==0]
    df_abnormal = dataset.data[y!=0]
    cols = dataset.data.drop(columns=['label']).columns
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    
    for c in cols:
        fig = plt.figure(figsize=(14,5))
        fig.suptitle('distributions of feature ' + c)
        df_normal[c].hist(legend=True, color='b', alpha=alpha, log=True)
        df_abnormal[c].hist(legend=True, color='r', alpha=alpha, log=True)
        plt.savefig(fname=f'{output_dir}/analysis/hist_{c}.png', bbox_inches='tight')
        plt.close()

def get_scores(path: str):
    scores, algo, ds = [], [], []
    #get local directories 1-level deep
    _, dirs, _ = next(os.walk(path))
    for d in dirs:
        for file in os.listdir(f"{path}/{d}"):
            if file.endswith('.txt'):
                lines = open(f"{path}/{d}/{file}").readlines()
                for line in lines:
                    if 'macro avg' in line:
                        scores.append(float(line.split()[-2]))
                        algo.append(file[:-4].lower())
                        ds.append(d.split('dataset=')[1].split(',')[0])
                        
    results = {'F1-macro':scores,
              'Dataset':ds,
              'Algorithm': algo,
              'set':["train","test"]*int(len(scores)/2)}
    
    return pd.DataFrame(results)

def plot_overview(df: pd.DataFrame, path: str = None):
    f, axs = plt.subplots(1, len(df.Dataset.unique()), sharey=True)
    subplots = []
    if len(df.Dataset.unique()) == 1:
        dataset=df.Dataset.values[0]
        p = sns.barplot(data=df[df.Dataset == dataset], x='Algorithm', y='F1-macro', hue='set', ax=axs)
        p.legend_.remove()
        p.set_title(dataset.upper())
        for container in p.containers:
            p.bar_label(container)
            
        axs.tick_params(axis='x', rotation=90)
        
    else:
        for idx , dataset in enumerate(df.Dataset.unique()):
            subplots.append(sns.barplot(data=df[df.Dataset == dataset], x='Algorithm', y='F1-macro', hue='set', ax=axs[idx]))
            subplots[idx].legend_.remove()
            subplots[idx].set_title(dataset.upper())
            for container in subplots[idx].containers:
                subplots[idx].bar_label(container)
        
        for ax in axs:
            ax.tick_params(axis='x', rotation=90)
    
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
    f.tight_layout()
    if path:
        plt.savefig(f"{path}/overview-scores.png")
    else:
        plt.show()
