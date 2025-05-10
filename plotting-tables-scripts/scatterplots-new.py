import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATE = "Mar-18-2025"
DATASET = "MMLU"  # Can be "EWOK", "COMPS", "BABI", "ARITH", or "MMLU"

titledataset = {
    "EWOK": "EWoK",
    "COMPS": "COMPS",
    "BABI": "bAbI",
    "ARITH": "Arithmetic",
    "MMLU": "MMLU"}

titleshot = {
    "zeroshot": "Question-only",
    "fewshot": "Instruction + few-shot + Q",
    "instronly": "Instruction + Q"
}

def load_metrics_from_json(shot_type, model_family):
    """Load metrics from the new JSON file format."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Constructing the JSON file path based on the new naming convention
    json_file_path = os.path.join(parent_dir, 'plotting-tables-scripts', 'overall_model_results', f'{shot_type}_{model_family.lower()}_{DATASET.lower()}.json')
    
    if not os.path.exists(json_file_path):
        print(f"File not found: {json_file_path}")
        return {}
    
    with open(json_file_path, 'r') as f:
        return json.load(f)

def create_combined_performance_plot(model_families, domains, shot_type):
    # Set different figure sizes based on whether we need a legend
    # The plot area will be the same size for all plot types
    PLOT_WIDTH = 14  # Width for the plot area only
    LEGEND_WIDTH = 4  # Width for the legend area
    PLOT_HEIGHT = 12  # Height for all plots
    
    if shot_type == "fewshot":
        # For fewshot, include space for the legend
        plt.figure(figsize=(PLOT_WIDTH + LEGEND_WIDTH, PLOT_HEIGHT))
    else:
        # For zeroshot and instronly, no legend needed
        plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    
    family_colors = {
        'Falcon' : ['#90CAF9', '#42A5F5', '#1E88E5', '#0D47A1', '#082A5F'],
        'MPT' : ['#EF9A9A', '#E57373', '#E53935', '#B71C1C', '#6D1010'],
        'Qwen' : ['#A5D6A7', '#66BB6A', '#43A047', '#1B5E20', '#0F3512'],
        'Llama' : ['#CE93D8', '#AB47BC', '#8E24AA', '#4A148C', '#2A0B50']
    }
    
    # Track if any data was plotted
    data_plotted = False
    
    for model_family, models in model_families.items():
        shades = family_colors[model_family]
        
        # Load all metrics for this model family and shot type
        all_metrics = load_metrics_from_json(shot_type, model_family)
        
        if not all_metrics:
            print(f"No data for {model_family} in {shot_type} mode")
            continue
        
        for idx, model in enumerate(models):
            # Ensure we don't go out of bounds with the shades
            if idx >= len(shades):
                shade_idx = len(shades) - 1
            else:
                shade_idx = idx
                
            plain_accuracies = []
            plain_biases = []
            kfold_accuracies = []
            kfold_biases = []
            
            for domain in domains:
                if domain in all_metrics and model in all_metrics[domain]:
                    model_metrics = all_metrics[domain][model]
                    
                    # Debug print to check values
                    print(f"Found data for {model} in {domain}: {model_metrics}")
                    
                    # Extract metrics directly from JSON
                    plain_accuracies.append(model_metrics['plain_acc'])
                    kfold_accuracies.append(model_metrics['kfold_acc'])
                    
                    # Handle the different bias fields based on dataset
                    if DATASET == "MMLU":
                        # For MMLU, use rstd values as bias
                        plain_biases.append(model_metrics['plain_rstd'])
                        kfold_biases.append(model_metrics['kfold_rstd'])
                    else:
                        # For other datasets, use the bias fields directly
                        plain_biases.append(model_metrics['plain_bias'])
                        kfold_biases.append(model_metrics['kfold_bias'])
            
            if plain_accuracies and kfold_accuracies:
                plain_mean_acc = np.mean(plain_accuracies)
                plain_mean_bias = np.mean(plain_biases)
                kfold_mean_acc = np.mean(kfold_accuracies)
                kfold_mean_bias = np.mean(kfold_biases)
                
                # Debug print means
                print(f"Plotting {model} with plain means: ({plain_mean_bias}, {plain_mean_acc})")
                print(f"Plotting {model} with kfold means: ({kfold_mean_bias}, {kfold_mean_acc})")
                
                # Determine if we need to add to legend
                # Only add to legend for fewshot plots
                if shot_type == "fewshot":
                    plt.scatter(plain_mean_bias, plain_mean_acc, 
                             marker='o', s=200, c=[shades[shade_idx]], 
                             label=f'{model}')
                else:
                    plt.scatter(plain_mean_bias, plain_mean_acc, 
                             marker='o', s=200, c=[shades[shade_idx]])
                
                plt.scatter(kfold_mean_bias, kfold_mean_acc, 
                          marker='x', linewidths=3, s=200, c=[shades[shade_idx]])
                plt.plot([plain_mean_bias, kfold_mean_bias], 
                         [plain_mean_acc, kfold_mean_acc], 
                         c=shades[shade_idx], linestyle=':', alpha=1, linewidth=4)
                
                data_plotted = True
                
    if not data_plotted:
        print("No data was plotted. Check your JSON files and paths.")
        return False
    
    if DATASET == "MMLU":
        plt.xlabel('RStd Value', fontsize=40, labelpad=10)
        plt.xlim(0, 1)
        plt.ylim(0.25, 0.75)
        plt.yticks(np.arange(0.25, 0.8, 0.1))
    else:
        plt.xlabel('Yes-No Bias Score', fontsize=40, labelpad=10)
        plt.xlim(-1, 1)
        # set xticks to -1, -0.5, 0, 0.5, 1
        plt.xticks(np.arange(-1, 1.1, 0.5))
        plt.ylim(0.4, 1)
        plt.axvline(x=0, color='black', linestyle='--', alpha=1)
    
    plt.ylabel('Accuracy', fontsize=40, labelpad=10)
    plt.title(f'{titledataset[DATASET]}, {titleshot[shot_type]}', fontsize=40, y=1.025)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=35, y=-0.02)
    plt.yticks(fontsize=35, x=-0.01)
    
    # Only add legend for fewshot plots
    if shot_type == "fewshot":
        legend = plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=25)
        legend_frame = legend.get_frame()
        legend_frame.set_width(LEGEND_WIDTH * 75)  # Adjust width of legend
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = f'outputs/{DATE}-new/{shot_type}'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f'{output_dir}/{shot_type}_{DATASET}_scatter.png'
    print(f"Saving plot to: {output_file}")
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
if DATASET == "EWOK":
    domains = ["social_interactions", "social_properties", "material_dynamics", "social_relations", 
               "quantitative_properties", "physical_dynamics", "physical_interactions", "material_properties", 
               "physical_relations", "spatial_relations", "agent_properties"]
elif DATASET == "COMPS":
    domains = ["comps"]
elif DATASET == "BABI":
    domains = ["babi"]
elif DATASET == "ARITH":
    domains = ["arith"]
elif DATASET == "MMLU":
    domains = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
        "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
        "college_medicine", "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
        "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
        "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
        "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
    ]

# Generate combined plots
for shot_type in ["zeroshot", "fewshot", "instronly"]:
    success = create_combined_performance_plot(model_families, domains, shot_type)
    if not success:
        print("IMPORTANT: No plot was generated. Check that your JSON files exist at the expected location.")
        print("Expected path format: parent_dir/plotting-tables-scripts/results/zeroshot_<model_family>_ewok.json")