import os
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats
import math

# Define model pairs (base model and its instruction-tuned counterpart)
model_pairs = {
    'Falcon': [('Falcon3-3B-Base', 'Falcon3-3B-Instruct'), 
               ('Falcon3-10B-Base', 'Falcon3-10B-Instruct')],
    'MPT': [('mpt-7b', 'mpt-7b-chat'), 
            ('mpt-30b', 'mpt-30b-chat')],
    'Qwen': [('Qwen1.5-7B', 'Qwen1.5-7B-Chat'), 
             ('Qwen1.5-32B', 'Qwen1.5-32B-Chat')],
    'Llama': [('Llama-2-7b-hf', 'Llama-2-7b-chat-hf'), 
              ('Llama-2-13b-hf', 'Llama-2-13b-chat-hf')]
}

# Define shot types and datasets
shot_types = ['zeroshot', 'instronly', 'fewshot']
datasets = ['ewok', 'comps', 'babi', 'arith', 'mmlu']

# Path to results directory
results_dir = 'results'

# Function to interpret Cohen's d
def interpret_cohens_d(d):
    if d is None or math.isnan(d):
        return 'N/A'
    # Based on Cohen's guidelines (taking absolute value for interpretation)
    d_abs = abs(d)
    if d_abs < 0.2:
        return 'Negligible'
    elif d_abs < 0.5:
        return 'Small'
    elif d_abs < 0.8:
        return 'Medium'
    else:
        return 'Large'

# Initialize results structure
results = {}

# Process each results file
for shot_type in shot_types:
    results[shot_type] = {}
    
    for dataset in datasets:
        # Initialize dataset results
        results[shot_type][dataset] = {
            'bias_diffs': [],  # Store all bias differences
            'base_biases_abs': [],  # Store absolute values of base biases
            'inst_biases_abs': [],  # Store absolute values of instruction-tuned biases
            'pairs': []  # Store detailed pairs info
        }
        
        # Process each model family
        for family in model_pairs.keys():
            # Construct filename
            filename = f"{shot_type}_{family.lower()}_{dataset}.json"
            filepath = os.path.join(results_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Warning: File {filepath} not found, skipping...")
                continue
                
            # Load the JSON data
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Process each domain in the file
            for domain, models in data.items():
                # Process each model pair in this family
                for base_model, inst_model in model_pairs[family]:
                    if base_model in models and inst_model in models:
                        if dataset == "mmlu":
                            base_bias = models[base_model].get('plain_rstd')
                            inst_bias = models[inst_model].get('plain_rstd')
                        else:
                            base_bias = models[base_model].get('plain_bias')
                            inst_bias = models[inst_model].get('plain_bias')
                        
                        if base_bias is not None and inst_bias is not None:
                            # Calculate absolute values since bias is -1 to 1 with 0 meaning no bias
                            base_bias_abs = abs(base_bias)
                            inst_bias_abs = abs(inst_bias)
                            
                            # Calculate bias difference (negative means less bias in inst model)
                            bias_diff = inst_bias_abs - base_bias_abs
                            
                            # Store pair details
                            results[shot_type][dataset]['pairs'].append({
                                'family': family,
                                'domain': domain,
                                'base_model': base_model,
                                'inst_model': inst_model,
                                'base_bias': base_bias,
                                'inst_bias': inst_bias,
                                'base_bias_abs': base_bias_abs,
                                'inst_bias_abs': inst_bias_abs,
                                'bias_diff': bias_diff
                            })
                            
                            # Store for statistical calculations
                            results[shot_type][dataset]['bias_diffs'].append(bias_diff)
                            results[shot_type][dataset]['base_biases_abs'].append(base_bias_abs)
                            results[shot_type][dataset]['inst_biases_abs'].append(inst_bias_abs)

# Calculate statistics and create summary table
summary_data = []

for shot_type in shot_types:
    for dataset in datasets:
        dataset_results = results[shot_type][dataset]
        bias_diffs = dataset_results['bias_diffs']
        
        if not bias_diffs:
            continue  # Skip if no data
            
        # Calculate mean bias difference
        mean_diff = np.mean(bias_diffs)
        
        # Calculate paired Cohen's d (d_z)
        std_diff = np.std(bias_diffs, ddof=1)  # Sample standard deviation
        if std_diff > 0:
            cohens_d = mean_diff / std_diff
        else:
            cohens_d = float('nan')
        
        # Paired t-test to test if mean difference is significantly different from 0
        t_stat, p_value = stats.ttest_1samp(bias_diffs, 0)
        
        # Mean absolute bias for base and instruction models
        mean_base_bias_abs = np.mean(dataset_results['base_biases_abs'])
        mean_inst_bias_abs = np.mean(dataset_results['inst_biases_abs'])
        
        # Percent reduction in absolute bias
        if mean_base_bias_abs > 0:
            percent_reduction = ((mean_base_bias_abs - mean_inst_bias_abs) / mean_base_bias_abs) * 100
        else:
            percent_reduction = float('nan')
        
        # Add to summary data
        summary_data.append({
            'Shot Type': shot_type,
            'Dataset': dataset,
            'Sample Size': len(bias_diffs),
            'Mean Abs Bias (Base)': mean_base_bias_abs,
            'Mean Abs Bias (Inst)': mean_inst_bias_abs,
            'Mean Bias Diff': mean_diff,
            'Percent Reduction': percent_reduction,
            'Cohen\'s d': cohens_d,
            'Effect Size': interpret_cohens_d(cohens_d),
            'p-value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

# Create a pandas DataFrame for the summary
summary_df = pd.DataFrame(summary_data)

# Generate detailed pairs data
pairs_data = []
for shot_type in shot_types:
    for dataset in datasets:
        for pair in results[shot_type][dataset]['pairs']:
            pairs_data.append({
                'Shot Type': shot_type,
                'Dataset': dataset,
                'Domain': pair['domain'],
                'Family': pair['family'],
                'Base Model': pair['base_model'],
                'Inst Model': pair['inst_model'],
                'Base Bias': pair['base_bias'],
                'Inst Bias': pair['inst_bias'],
                'Base Bias (Abs)': pair['base_bias_abs'],
                'Inst Bias (Abs)': pair['inst_bias_abs'],
                'Bias Diff': pair['bias_diff']
            })

detailed_df = pd.DataFrame(pairs_data)

# Display the summary table
print("\n=== SUMMARY OF ABSOLUTE BIAS DIFFERENCES (INSTRUCTION-TUNED - BASE) ===")
print("(Negative values indicate instruction-tuned models have LESS bias)")
print(summary_df.to_string(index=False))

# Save results to CSV
summary_df.to_csv('bias_analysis_summary.csv', index=False)
detailed_df.to_csv('bias_analysis_detailed.csv', index=False)

print("\nAnalysis complete. Results saved to 'bias_analysis_summary.csv' and 'bias_analysis_detailed.csv'")

# Print interpretation
print("\n=== INTERPRETATION ===")
print("A negative Mean Bias Diff indicates that instruction-tuned models have LESS bias than")
print("their base model counterparts (i.e., bias values closer to zero).")
print("A positive Mean Bias Diff indicates that instruction-tuned models have MORE bias.")
print("\nCohen's d interpretation:")
print("  < 0.2: Negligible effect")
print("  0.2 - 0.5: Small effect")
print("  0.5 - 0.8: Medium effect")
print("  > 0.8: Large effect")
print("\nThe p-value indicates whether the difference is statistically significant (α = 0.05).")
print("The Percent Reduction shows how much the absolute bias decreased in instruction-tuned models.")