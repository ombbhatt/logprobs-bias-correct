import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

DATE = "Mar-18-2025"

def load_metrics_from_json(dataset, shot_type, model_family):
    """Load metrics from the new JSON file format."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Constructing the JSON file path based on the new naming convention
    json_file_path = os.path.join(parent_dir, 'plotting-tables-scripts', 'results', f'{shot_type}_{model_family.lower()}_{dataset.lower()}.json')
    
    if not os.path.exists(json_file_path):
        print(f"File not found: {json_file_path}")
        return {}
    
    with open(json_file_path, 'r') as f:
        return json.load(f)

def create_combined_heatmap_plot(model_families, dataset_domains):
    # Exclude MMLU as specified
    datasets = ["EWOK", "COMPS", "BABI", "ARITH"]
    dataset_names = ["EWoK", "COMPS", "bAbI", "Arith."]
    
    # Gather all model names across families
    all_models = []
    for models in model_families.values():
        all_models.extend(models)
    
    # Define data structure for heatmap data
    heatmap_data = {}
    for shot_type in shot_types:
        heatmap_data[shot_type] = {
            'plain': np.zeros((len(datasets), len(all_models))),
            'kfold': np.zeros((len(datasets), len(all_models)))
        }
    
    # Fill heatmap data with bias/accuracy values
    for shot_type in shot_types:
        for d_idx, dataset in enumerate(datasets):
            domains = dataset_domains[dataset]
            
            col_idx = 0
            for family, models in model_families.items():
                all_metrics = load_metrics_from_json(dataset, shot_type, family)
                
                if not all_metrics:
                    print(f"No data for {family} in {dataset} for {shot_type}")
                    col_idx += len(models)
                    continue
                
                for model in models:
                    # Initialize with NaN to handle missing data
                    plain_biases = []
                    kfold_biases = []
                    
                    for domain in domains:
                        if domain in all_metrics and model in all_metrics[domain]:
                            model_metrics = all_metrics[domain][model]
                            
                            # Use bias fields directly
                            plain_biases.append(model_metrics['plain_bias'])
                            kfold_biases.append(model_metrics['kfold_bias'])
                    
                    if plain_biases and kfold_biases:
                        heatmap_data[shot_type]['plain'][d_idx, col_idx] = np.mean(plain_biases)
                        heatmap_data[shot_type]['kfold'][d_idx, col_idx] = np.mean(kfold_biases)
                    else:
                        print("ALERT WE HAVE NAN DATA")
                        heatmap_data[shot_type]['plain'][d_idx, col_idx] = np.nan
                        heatmap_data[shot_type]['kfold'][d_idx, col_idx] = np.nan
                    
                    col_idx += 1
    
    # Transpose data for the new layout (16x3 instead of 3x16)
    for shot_type in shot_types:
        heatmap_data[shot_type]['plain'] = heatmap_data[shot_type]['plain'].T
        heatmap_data[shot_type]['kfold'] = heatmap_data[shot_type]['kfold'].T
    
    fig = plt.figure(figsize=(40, 20))
    gs = GridSpec(2, 3, height_ratios=[1, 1])
    
    # Set up axes for each heatmap pair side by side
    ax_left_top = fig.add_subplot(gs[0, 0])      # zeroshot top
    ax_left_bottom = fig.add_subplot(gs[1, 0])   # zeroshot bottom
    
    ax_middle_top = fig.add_subplot(gs[0, 1])    # instronly top
    ax_middle_bottom = fig.add_subplot(gs[1, 1]) # instronly bottom
    
    ax_right_top = fig.add_subplot(gs[0, 2])     # fewshot top
    ax_right_bottom = fig.add_subplot(gs[1, 2])  # fewshot bottom
    
    cmap = 'RdBu_r'
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    
    # Function to plot a single heatmap
    def plot_single_heatmap(ax, data, title, is_top=True):
        # Plot heatmap
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
        
        # Add title
        ax.set_title(title, fontsize=24, pad=10, fontweight="bold")
        
        # Add labels
        if is_top:
            # For top heatmaps - only show model labels
            ax.set_yticks(range(len(all_models)))
            ax.set_yticklabels(all_models, fontsize=20)
            ax.set_xticks([])  # Hide x-axis labels on top heatmap
        else:
            # For bottom heatmaps - show both model and dataset labels
            ax.set_yticks(range(len(all_models)))
            ax.set_yticklabels(all_models, fontsize=20)
            ax.set_xticks(range(len(datasets)))
            ax.set_xticklabels(dataset_names, fontsize=20, fontweight="bold", rotation=0, ha='center')
        
        # Add grid lines
        ax.set_yticks(np.arange(-0.5, len(all_models), 1), minor=True)
        ax.set_xticks(np.arange(-0.5, len(datasets), 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, aspect=40, pad=0.04, ticks=[-1, 0, 1])
        cbar.ax.tick_params(labelsize=18)
        
        # Add text annotations with bias values
        for i in range(data.shape[0]):  # model index
            for j in range(data.shape[1]):  # dataset index
                if not np.isnan(data[i, j]):
                    # Choose text color based on bias value
                    text_color = "white" if abs(data[i, j]) > 0.8 else "black"
                    ax.text(j, i, f"{data[i, j]:.2f}", 
                            ha="center", va="center", color=text_color, fontsize=18)
        
        return im
    
    # Plot each heatmap
    plot_single_heatmap(ax_left_top, heatmap_data['zeroshot']['plain'], 
                        'Question-only : Before Correction', True)
    plot_single_heatmap(ax_left_bottom, heatmap_data['zeroshot']['kfold'], 
                        'Question-only : After Correction', False)
    
    plot_single_heatmap(ax_middle_top, heatmap_data['instronly']['plain'], 
                        'Instruction + Q : Before Correction', True)
    plot_single_heatmap(ax_middle_bottom, heatmap_data['instronly']['kfold'], 
                        'Instruction + Q : After Correction', False)
    
    plot_single_heatmap(ax_right_top, heatmap_data['fewshot']['plain'], 
                        'Instruction + few-shot + Q : Before Correction', True)
    plot_single_heatmap(ax_right_bottom, heatmap_data['fewshot']['kfold'], 
                        'Instruction + few-shot + Q : After Correction', False)
    
    # Add spacing between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.15)
    
    # Create output directory
    output_dir = f'outputs/{DATE}-new'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save figure
    output_file = f'{output_dir}/composite_bias_heatmap_sidebyside.png'
    print(f"Saving heatmap visualization to: {output_file}")
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    return True

# Define all model families and their models
model_families = {
    'Falcon': ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"],
    'MPT': ["mpt-7b", "mpt-7b-chat", "mpt-30b", "mpt-30b-chat"],
    'Qwen': ["Qwen1.5-7B", "Qwen1.5-7B-Chat", "Qwen1.5-32B", "Qwen1.5-32B-Chat"],
    'Llama': ["Llama-2-7b-hf", "Llama-2-7b-chat-hf", "Llama-2-13b-hf", "Llama-2-13b-chat-hf"]
}

# Define domains based on dataset
dataset_domains = {
    "EWOK": ["social_interactions", "social_properties", "material_dynamics", "social_relations", 
             "quantitative_properties", "physical_dynamics", "physical_interactions", "material_properties", 
             "physical_relations", "spatial_relations", "agent_properties"],
    "COMPS": ["comps"],
    "BABI": ["babi"],
    "ARITH": ["arith"]
}

shot_types = ["zeroshot", "instronly", "fewshot"]

# Generate combined plots
success = create_combined_heatmap_plot(model_families, dataset_domains)
if not success:
    print("IMPORTANT: No plot was generated. Check that your JSON files exist at the expected location.")