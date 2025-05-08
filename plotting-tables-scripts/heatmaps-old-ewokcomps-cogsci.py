import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DATE = "Jan-21-2025"

def extract_metrics(last_row):
    metrics_str = last_row.split(',')
    accuracy = float(metrics_str[0].split(': ')[1])
    # print('accuracy: ' + str(accuracy))
    bias = float(metrics_str[5].split(': ')[1])
    # print('bias: ' + str(bias))
    # print(f'\n')
    return accuracy, bias

def get_metrics_for_model(base_path, model_family, model_name, domain):
    file_path = os.path.join(base_path, model_family, domain, f"{model_name}_results.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        last_row = df.iloc[-1, 0]
        return extract_metrics(last_row)
    return None

# Define all model families and their models
model_families = {
    'GPT2': ["gpt2"],
    'Falcon': ["Falcon3-10B-Base", "Falcon3-10B-Instruct", "Falcon3-Mamba-7B-Base", "Falcon3-Mamba-7B-Instruct"],
    'MPT': ["mpt-7b", "mpt-7b-chat", "mpt-7b-8k", "mpt-7b-8k-chat"],
    'Qwen': ["Qwen1.5-7B", "Qwen1.5-7B-Chat", "Qwen1.5-14B", "Qwen1.5-14B-Chat"],
    'Olmo': ["Olmo-2-1124-7B", "Olmo-2-1124-7B-Instruct", "Olmo-2-1124-13B", "Olmo-2-1124-13B-Instruct"]
}

dict_accs = {}
dict_biases = {}

def create_dataframe(dataset, shot_type):

    if dataset == "EWOK":
    # List of domains
        domains = ["social_interactions", "social_properties", "material_dynamics", 
            "social_relations", "quantitative_properties", "physical_dynamics", 
            "physical_interactions", "material_properties", "physical_relations", 
            "spatial_relations", "agent_properties"]
    elif dataset == "COMPS":
        domains = ["comps"]

    for model_family, models in model_families.items():

        for idx, model in enumerate(models):
            plain_accuracies = []
            plain_biases = []
            bos_accuracies = []
            bos_biases = []
            kfold_accuracies = []
            kfold_biases = []
        
            for domain in domains:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                
                # print(f"printing values for {shot_type} for {model} for {domain} for plain: ")
                plain_metrics = get_metrics_for_model(os.path.join(parent_dir, f"outputs/{DATE}/{shot_type}/{dataset}/plain"), model_family, model, domain)
                # print(f"printing values for {shot_type} for {model} for {domain} bos: ")
                bos_metrics = get_metrics_for_model(os.path.join(parent_dir, f"outputs/{DATE}/{shot_type}/{dataset}/bos"), model_family, model, domain)
                # print(f"printing values for {shot_type} for {model} for {domain} for kfold: ")
                kfold_metrics = get_metrics_for_model(os.path.join(parent_dir, f"outputs/{DATE}/{shot_type}/{dataset}/kfold"), model_family, model, domain)
                
                if plain_metrics:
                    accuracy, bias = plain_metrics
                    plain_accuracies.append(accuracy)
                    plain_biases.append(bias)

                if bos_metrics:
                    accuracy, bias = bos_metrics
                    bos_accuracies.append(accuracy)
                    bos_biases.append(bias)

                if kfold_metrics:
                    accuracy, bias = kfold_metrics
                    kfold_accuracies.append(accuracy)
                    kfold_biases.append(bias)

            if plain_accuracies and bos_accuracies and kfold_accuracies:
                plain_mean_acc = np.mean(plain_accuracies)
                plain_mean_bias = np.mean(plain_biases)

                bos_mean_acc = np.mean(bos_accuracies)
                bos_mean_bias = np.mean(bos_biases)

                kfold_mean_acc = np.mean(kfold_accuracies)
                kfold_mean_bias = np.mean(kfold_biases)

                dict_accs[model] = [plain_mean_acc, bos_mean_acc, kfold_mean_acc]
                dict_biases[model] = [plain_mean_bias, bos_mean_bias, kfold_mean_bias]

    df_biases = pd.DataFrame(dict_biases, index=['Base', 'Generic', 'Specific'])
    print(df_biases.head())
    return df_biases

zeroshot_ewok_df = create_dataframe("EWOK", "zeroshot")
fewshot_ewok_df = create_dataframe("EWOK", "fewshot")
zeroshot_comps_df = create_dataframe("COMPS", "zeroshot")
fewshot_comps_df = create_dataframe("COMPS", "fewshot")

dataframes = {
    'zeroshot': {
        'COMPS-YNQ': zeroshot_comps_df,
        'EWoK-YNQ': zeroshot_ewok_df
    },
    'fewshot': {
        'COMPS-YNQ': fewshot_comps_df,
        'EWoK-YNQ': fewshot_ewok_df
    }
}

def create_procedure_heatmaps(dataframes, save_path=None):
    """
    Create 6 heatmaps arranged in 2 vertical stacks for comparing datasets across models.
    
    Params:
    dataframes : dict
        Dictionary containing DataFrames with structure:
        {
            'zeroshot': {
                'COMPS': pd.DataFrame,  # DataFrame for dataset COMPS, zeroshot
                'EWOK': pd.DataFrame   # DataFrame for dataset EWOK, zeroshot
            },
            'fewshot': {
                'COMPS': pd.DataFrame,  # DataFrame for dataset COMPS, fewshot
                'EWOK': pd.DataFrame   # DataFrame for dataset EWOK, fewshot
            }
        }
    save_path : str, optional
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(40, 13))
    plt.subplots_adjust(hspace=0.1)  
    plt.subplots_adjust(wspace=-0.2)  
    
    # Define procedures and datasets
    procedures = ['Base', 'Generic', 'Specific']
    datasets = ['COMPS-YNQ', 'EWoK-YNQ']
    prompt_types = ['zeroshot', 'instronly', 'fewshot']
    
    # cmap = sns.diverging_palette(240, 10, as_cmap=True)  # Blue to Red
    
    # Define model family spans for vertical lines
    model_families = {
        'Falcon': 4,
        'MPT': 4,
        'Qwen': 4,
        'Llama': 4
    }
    family_boundaries = np.cumsum([0] + list(model_families.values()))
    
    for proc_idx, proc in enumerate(procedures):
        for prompt_idx, prompt in enumerate(prompt_types):
            # Create data for this heatmap
            heatmap_data = []
            for dataset in datasets:
                row_data = dataframes[prompt][dataset].loc[proc]
                heatmap_data.append(row_data)
            
            heatmap_df = pd.DataFrame(heatmap_data, 
                                    index=datasets,
                                    columns=dataframes[prompt]['COMPS'].columns)
            
            ax = axes[proc_idx, prompt_idx]
            hm = sns.heatmap(heatmap_df, 
                       annot=True,
                       fmt='.2f',
                       ax=ax,
                       cmap='RdBu_r', 
                       vmin=-1.0,
                       vmax=1.0,
                       center=0,
                       cbar_kws={'pad': 0.025},
                       xticklabels=True if proc_idx == 2 else False,  # Only show x labels for bottom row
                       yticklabels=True)
            
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=22)
            ax.figure.axes[-1].yaxis.label.set_size(20)

            # set annotation font size:
            for t in ax.texts: t.set_fontsize(21)
            
            # vertical lines between model families
            for boundary in family_boundaries[1:-1]:
                ax.axvline(x=boundary, color='black', linewidth=1)
                ax.axvline(x=boundary, color='white', linewidth=3)
                ax.axvline(x=boundary, color='black', linewidth=1)
            
            ax.set_title(f"{proc}", fontsize='28', fontweight='bold')

            plt.setp(ax.get_yticklabels(), rotation=0, fontsize='21', fontweight='bold')
            
            if proc_idx == 2:  # Bottom row
                plt.setp(ax.get_xticklabels(), rotation=50, ha='right', fontsize='26')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

current_dir = os.path.dirname(os.path.abspath(__file__))            
parent_dir = os.path.dirname(current_dir)
create_procedure_heatmaps(dataframes, save_path=os.path.join(parent_dir, f'outputs/Jan-22-2025/heatmaps.png'))