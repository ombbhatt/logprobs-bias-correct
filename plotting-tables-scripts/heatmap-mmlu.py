import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

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

def normalize_recalls(recalls):
    """Normalize recall values to sum to 1."""
    total = sum(recalls.values())
    if total == 0:
        # Avoid division by zero
        return {k: 0.25 for k in recalls}
    return {k: v / total for k, v in recalls.items()}

def get_dominant_option_color(normalized_recalls):
    """Get color based on the dominant option and its intensity based on deviation from 0.25."""
    # Base colors for each option
    color_map = {
        'A': np.array([1.0, 0.0, 0.0]),  # Red
        'B': np.array([0.0, 1.0, 0.0]),  # Green
        'C': np.array([0.0, 0.0, 1.0]),  # Blue
        'D': np.array([1.0, 1.0, 0.0])   # Yellow
    }
    
    # Find the dominant option
    dominant_option = max(normalized_recalls.items(), key=lambda x: x[1])
    dominant_key = dominant_option[0]
    dominant_value = dominant_option[1]
    
    # Calculate intensity based on deviation from 0.25 (random chance)
    # Scale from 0.1 (at 0.25) to 1.0 (at 1.0)
    min_intensity = 0.01  # Faint but visible at 0.25
    
    # Linear scaling from min_intensity at 0.25 to 1.0 at 1.0
    intensity = min_intensity + (dominant_value - 0.25) * (1.0 - min_intensity) / 0.75
    
    # Cap intensity at min_intensity for values below 0.25
    intensity = max(min_intensity, intensity)
    
    # Get base color for dominant option and adjust intensity
    base_color = color_map[dominant_key]
    
    # Use white as background and blend with base color according to intensity
    final_color = (1.0 - intensity) * np.array([1.0, 1.0, 1.0]) + intensity * base_color
    
    return final_color, dominant_key, dominant_value

def create_mmlu_heatmap_plot(model_families, dataset_domains):
    dataset = "MMLU"
    
    # Gather all model names across families
    all_models = []
    for models in model_families.values():
        all_models.extend(models)
    
    # Define data structure for heatmap data - store RGB color values and dominant option
    heatmap_data = {}
    for shot_type in shot_types:
        heatmap_data[shot_type] = {
            'plain': {'colors': np.zeros((1, len(all_models), 3)), 'dominant': {}},
            'kfold': {'colors': np.zeros((1, len(all_models), 3)), 'dominant': {}}
        }
    
    # Also store the raw normalized values for annotation
    normalized_values = {}
    for shot_type in shot_types:
        normalized_values[shot_type] = {
            'plain': {},
            'kfold': {}
        }
    
    # Fill heatmap data with color values
    for shot_type in shot_types:
        domains = dataset_domains[dataset]
        domain_count = {}  # Track how many domains contributed to each model
        
        col_idx = 0
        for family, models in model_families.items():
            all_metrics = load_metrics_from_json(dataset, shot_type, family)
            
            if not all_metrics:
                print(f"No data for {family} in {dataset} for {shot_type}")
                col_idx += len(models)
                continue
            
            for model in models:
                # Track aggregated recalls across domains
                plain_recalls_agg = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
                kfold_recalls_agg = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
                domain_count[col_idx] = 0
                
                for domain in domains:
                    if domain in all_metrics and model in all_metrics[domain]:
                        model_metrics = all_metrics[domain][model]
                        
                        if 'plain_recalls' in model_metrics and 'kfold_recalls' in model_metrics:
                            # Extract recall values for different options
                            plain_recalls = model_metrics['plain_recalls']
                            kfold_recalls = model_metrics['kfold_recalls']
                            
                            # Aggregate recalls across domains
                            for option in ['A', 'B', 'C', 'D']:
                                plain_recalls_agg[option] += plain_recalls.get(option, 0)
                                kfold_recalls_agg[option] += kfold_recalls.get(option, 0)
                            
                            domain_count[col_idx] += 1
                
                # Calculate average recalls across domains
                if domain_count[col_idx] > 0:
                    for option in ['A', 'B', 'C', 'D']:
                        plain_recalls_agg[option] /= domain_count[col_idx]
                        kfold_recalls_agg[option] /= domain_count[col_idx]
                
                # Normalize and get dominant option colors
                normalized_plain = normalize_recalls(plain_recalls_agg)
                normalized_kfold = normalize_recalls(kfold_recalls_agg)
                
                # Store normalized values for annotations
                normalized_values[shot_type]['plain'][col_idx] = normalized_plain
                normalized_values[shot_type]['kfold'][col_idx] = normalized_kfold
                
                # Get colors based on dominant option and intensity
                plain_color, plain_dominant, plain_value = get_dominant_option_color(normalized_plain)
                kfold_color, kfold_dominant, kfold_value = get_dominant_option_color(normalized_kfold)
                
                # Store colors and dominant options
                heatmap_data[shot_type]['plain']['colors'][0, col_idx] = plain_color
                heatmap_data[shot_type]['plain']['dominant'][col_idx] = (plain_dominant, plain_value)
                
                heatmap_data[shot_type]['kfold']['colors'][0, col_idx] = kfold_color
                heatmap_data[shot_type]['kfold']['dominant'][col_idx] = (kfold_dominant, kfold_value)
                
                col_idx += 1
    
    # Set up the figure
    fig = plt.figure(figsize=(40, 5))
    gs = GridSpec(2, 3, height_ratios=[1, 1])
    
    # Set up axes for each heatmap pair side by side
    ax_left_top = fig.add_subplot(gs[0, 0])      # zeroshot top
    ax_left_bottom = fig.add_subplot(gs[1, 0])   # zeroshot bottom
    
    ax_middle_top = fig.add_subplot(gs[0, 1])    # instronly top
    ax_middle_bottom = fig.add_subplot(gs[1, 1]) # instronly bottom
    
    ax_right_top = fig.add_subplot(gs[0, 2])     # fewshot top
    ax_right_bottom = fig.add_subplot(gs[1, 2])  # fewshot bottom
    
    # Function to plot a single heatmap with dominant option colors
    def plot_dominant_heatmap(ax, data, normalized_data, dominant_data, title, is_top=True):
        # Create a new array for imshow that is the right shape
        img_data = np.zeros((data.shape[0], data.shape[1], 4))
        
        # Set RGB values from our data
        img_data[:, :, 0:3] = data
        
        # Set alpha channel to 1 (fully opaque)
        img_data[:, :, 3] = 1
        
        # Display the image
        ax.imshow(img_data, aspect='auto')
        
        # Add title
        ax.set_title(title, fontsize=24, pad=10, fontweight="bold")
        
        # Add labels
        ax.set_yticks([])  # No y-ticks for single row
        
        if is_top:
            # For top heatmaps - hide x-axis labels
            ax.set_xticks([])
        else:
            # For bottom heatmaps - show dataset labels (MMLU)
            ax.set_xticks(range(len(all_models)))
            ax.set_xticklabels(all_models, fontsize=19, rotation=90, ha='right')
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, len(all_models), 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        
        # Add text annotations with normalized values and dominant option
        for j in range(data.shape[1]):  # model index
            if j in normalized_data:
                values = normalized_data[j]
                dominant_info = dominant_data[j]
                
                # Choose text color based on background brightness
                brightness = np.mean(data[0, j])
                text_color = "black" if brightness > 0.5 else "white"
                
                # Create annotation with dominant option highlighted
                # annotation = f"Dom: {dominant_info[0]} ({dominant_info[1]:.2f})\n"
                annotation = f""
                for option in ['A', 'B', 'C', 'D']:
                    # if option == dominant_info[0]:
                        # annotation += f"→{option}: {values[option]:.2f}\n"
                    # else:
                        annotation += f"{option} {values[option]:.2f}\n"
                annotation = annotation[:-1]  # Remove last newline
                
                ax.text(j, 0, annotation, 
                        ha="center", va="center", color=text_color, fontsize=12)
        
        return ax
    
    # Plot each heatmap
    plot_dominant_heatmap(ax_left_top, 
                         heatmap_data['zeroshot']['plain']['colors'],
                         normalized_values['zeroshot']['plain'], 
                         heatmap_data['zeroshot']['plain']['dominant'],
                         'Question-only : Before Correction', True)
    
    plot_dominant_heatmap(ax_left_bottom, 
                         heatmap_data['zeroshot']['kfold']['colors'],
                         normalized_values['zeroshot']['kfold'], 
                         heatmap_data['zeroshot']['kfold']['dominant'],
                         'Question-only : After Correction', False)
    
    plot_dominant_heatmap(ax_middle_top, 
                         heatmap_data['instronly']['plain']['colors'],
                         normalized_values['instronly']['plain'], 
                         heatmap_data['instronly']['plain']['dominant'],
                         'Instruction + Q : Before Correction', True)
    
    plot_dominant_heatmap(ax_middle_bottom, 
                         heatmap_data['instronly']['kfold']['colors'],
                         normalized_values['instronly']['kfold'], 
                         heatmap_data['instronly']['kfold']['dominant'],
                         'Instruction + Q : After Correction', False)
    
    plot_dominant_heatmap(ax_right_top, 
                         heatmap_data['fewshot']['plain']['colors'],
                         normalized_values['fewshot']['plain'], 
                         heatmap_data['fewshot']['plain']['dominant'],
                         'Instruction + few-shot + Q : Before Correction', True)
    
    plot_dominant_heatmap(ax_right_bottom, 
                         heatmap_data['fewshot']['kfold']['colors'],
                         normalized_values['fewshot']['kfold'], 
                         heatmap_data['fewshot']['kfold']['dominant'],
                         'Instruction + few-shot + Q : After Correction', False)
    
    # Create a custom legend with the option colors
    base_colors = {
        'A': [1.0, 0.0, 0.0],  # Red
        'B': [0.0, 1.0, 0.0],  # Green
        'C': [0.0, 0.0, 1.0],  # Blue
        'D': [1.0, 1.0, 0.0]   # Yellow
    }
    
    # Create patches for each option and its color
    legend_patches = [
        mpatches.Patch(color=base_colors['A'], label='A'),
        mpatches.Patch(color=base_colors['B'], label='B'),
        mpatches.Patch(color=base_colors['C'], label='C'),
        mpatches.Patch(color=base_colors['D'], label='D')
    ]
    
    # Create gradient patches to explain color intensity
    gradient_colors = []
    for i in range(5):
        # Create gradients from 0.1 to 1.0 intensity (applied to red for demonstration)
        intensity = 0.1 + i * 0.225
        color = (1.0 - intensity) * np.array([1.0, 1.0, 1.0]) + intensity * np.array([1.0, 0.0, 0.0])
        gradient_colors.append(color)
    
    gradient_patches = [
        mpatches.Patch(color=gradient_colors[0], label='Dominant value: 0.25 (Random)'),
        mpatches.Patch(color=gradient_colors[2], label='Dominant value: ~0.5'),
        mpatches.Patch(color=gradient_colors[4], label='Dominant value: ~1.0')
    ]
    
    all_patches = legend_patches + gradient_patches
    
    # Add legend to the plot
    fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=5, fontsize=20)
    # fig.legend(handles=all_patches, loc='upper center', 
    #            bbox_to_anchor=(0.5, 0.1), ncol=7, fontsize=18)
    
    # # Add explanation
    # plt.figtext(0.5, 0.05, 
    #             "Colors represent the dominant response option (A,B,C,D).\n"
    #             "Color intensity shows how strongly the option dominates (faint = near random chance at 0.25, intense = strong preference).",
    #             ha='center', fontsize=18, bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    
    # Add spacing between subplots
    # plt.subplots_adjust(wspace=0.4, hspace=0.15, bottom=0.2)
    plt.subplots_adjust(wspace=0.06, hspace=0.4, bottom=0.2)
    
    # Create output directory
    output_dir = f'outputs/{DATE}-new'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save figure
    output_file = f'{output_dir}/mmlu_dominant_option_heatmap.png'
    print(f"Saving MMLU dominant option heatmap to: {output_file}")
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

# Define domains for MMLU
dataset_domains = {
    "MMLU": [
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
}

shot_types = ["zeroshot", "instronly", "fewshot"]

# Generate combined plots
success = create_mmlu_heatmap_plot(model_families, dataset_domains)
if not success:
    print("IMPORTANT: No plot was generated. Check that your JSON files exist at the expected location.")