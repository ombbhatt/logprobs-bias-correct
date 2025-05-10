import os
import json
import glob
import pandas as pd
import numpy as np

# Define model families and shot types
model_families = {
    'Falcon': ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"],
    'MPT': ["mpt-7b", "mpt-7b-chat", "mpt-30b", "mpt-30b-chat"],
    'Qwen': ["Qwen1.5-7B", "Qwen1.5-7B-Chat", "Qwen1.5-32B", "Qwen1.5-32B-Chat"],
    'Llama': ["Llama-2-7b-hf", "Llama-2-7b-chat-hf", "Llama-2-13b-hf", "Llama-2-13b-chat-hf"]
}

shot_types = ['zeroshot', 'instronly', 'fewshot']
datasets = ['ewok', 'comps', 'blimp', 'babi', 'arith', 'mmlu']

# Path to results directory
results_dir = 'results'

# Initialize results structure for per-dataset, per-shot type analysis
dataset_shot_results = {}

# Initialize a separate structure for MMLU individual model analysis
# For each shot type, track accumulated values per model
mmlu_model_data = {
    shot_type: {
        model_name: {
            'plain_acc_sum': 0.0, 'kfold_acc_sum': 0.0,
            'plain_rstd_sum': 0.0, 'kfold_rstd_sum': 0.0,
            'domain_count': 0
        } for family in model_families.values() for model_name in family
    } for shot_type in shot_types
}

# Process all result files
for shot_type in shot_types:
    dataset_shot_results[shot_type] = {}
    
    for dataset in datasets:
        dataset_shot_results[shot_type][dataset] = {
            'plain_acc_values': [],
            'kfold_acc_values': [],
            'plain_bias_values': [],
            'kfold_bias_values': [],
            'model_count': 0
        }
        
        # Check all model families for this dataset and shot type
        for family in model_families.keys():
            filename = f"{shot_type}_{family.lower()}_{dataset}.json"
            filepath = os.path.join(results_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Warning: File {filepath} not found, skipping...")
                continue
            
            # Load the JSON data
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Process each domain and model
            for domain, models in data.items():
                for model_name in model_families[family]:
                    if model_name in models:
                        model_data = models[model_name]
                        
                        # Collect metrics
                        if 'plain_acc' in model_data:
                            dataset_shot_results[shot_type][dataset]['plain_acc_values'].append(model_data['plain_acc'])
                        if 'kfold_acc' in model_data:
                            dataset_shot_results[shot_type][dataset]['kfold_acc_values'].append(model_data['kfold_acc'])
                        
                        # Handle the different bias metrics for MMLU vs other datasets
                        if dataset == 'mmlu':
                            dataset_shot_results[shot_type][dataset]['plain_bias_values'].append(abs(model_data['plain_rstd']))
                            dataset_shot_results[shot_type][dataset]['kfold_bias_values'].append(abs(model_data['kfold_rstd']))
                            
                            # For MMLU, accumulate values for each model across all domains
                            if 'plain_acc' in model_data and 'kfold_acc' in model_data and 'plain_rstd' in model_data and 'kfold_rstd' in model_data:
                                plain_acc = model_data['plain_acc']
                                kfold_acc = model_data['kfold_acc']
                                plain_rstd = abs(model_data['plain_rstd'])
                                kfold_rstd = abs(model_data['kfold_rstd'])
                                
                                # Accumulate values for this model
                                mmlu_model_data[shot_type][model_name]['plain_acc_sum'] += plain_acc
                                mmlu_model_data[shot_type][model_name]['kfold_acc_sum'] += kfold_acc
                                mmlu_model_data[shot_type][model_name]['plain_rstd_sum'] += plain_rstd
                                mmlu_model_data[shot_type][model_name]['kfold_rstd_sum'] += kfold_rstd
                                mmlu_model_data[shot_type][model_name]['domain_count'] += 1
                        else:
                            dataset_shot_results[shot_type][dataset]['plain_bias_values'].append(abs(model_data['plain_bias']))
                            dataset_shot_results[shot_type][dataset]['kfold_bias_values'].append(abs(model_data['kfold_bias']))
                
                # Increment model count (only once per domain to avoid double counting)
                dataset_shot_results[shot_type][dataset]['model_count'] += 1

# Calculate the aggregate metrics for each dataset and shot type
summary_data = []

for shot_type in shot_types:
    for dataset in datasets:
        results = dataset_shot_results[shot_type][dataset]
        
        # Skip if no data
        if not results['plain_acc_values']:
            continue
            
        # Calculate averages
        avg_plain_acc = np.mean(results['plain_acc_values'])
        avg_kfold_acc = np.mean(results['kfold_acc_values']) if results['kfold_acc_values'] else np.nan
        avg_plain_bias = np.mean(results['plain_bias_values'])
        avg_kfold_bias = np.mean(results['kfold_bias_values']) if results['kfold_bias_values'] else np.nan
        
        # Calculate percentage differences
        acc_pct_diff = ((avg_kfold_acc - avg_plain_acc) / avg_plain_acc * 100) if not np.isnan(avg_plain_acc) and avg_plain_acc != 0 else np.nan
        bias_pct_diff = ((avg_kfold_bias - avg_plain_bias) / avg_plain_bias * 100) if not np.isnan(avg_plain_bias) and avg_plain_bias != 0 else np.nan
        
        # Add to summary data
        summary_data.append({
            'Shot Type': shot_type,
            'Dataset': dataset,
            'Avg Plain Accuracy': avg_plain_acc,
            'Avg KFold Accuracy': avg_kfold_acc,
            'Accuracy % Difference': acc_pct_diff,
            'Avg Plain Bias (Abs)': avg_plain_bias,
            'Avg KFold Bias (Abs)': avg_kfold_bias,
            'Bias % Difference': bias_pct_diff,
            'Sample Count': len(results['plain_acc_values'])
        })

# Convert to DataFrame and sort
summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values(['Shot Type', 'Dataset'])

# Format percentages and round decimals for better readability
summary_df['Accuracy % Difference'] = summary_df['Accuracy % Difference'].map('{:+.2f}%'.format)
summary_df['Bias % Difference'] = summary_df['Bias % Difference'].map('{:+.2f}%'.format)
summary_df['Avg Plain Accuracy'] = summary_df['Avg Plain Accuracy'].map('{:.4f}'.format)
summary_df['Avg KFold Accuracy'] = summary_df['Avg KFold Accuracy'].map('{:.4f}'.format)
summary_df['Avg Plain Bias (Abs)'] = summary_df['Avg Plain Bias (Abs)'].map('{:.4f}'.format)
summary_df['Avg KFold Bias (Abs)'] = summary_df['Avg KFold Bias (Abs)'].map('{:.4f}'.format)

# Save to CSV
summary_df.to_csv('dataset_shot_statistics.csv', index=False)

# Create MMLU individual model summaries
mmlu_individual_dfs = {}
for shot_type in shot_types:
    # Create a list to hold the summary for each model
    model_summaries = []
    
    for family_name, models in model_families.items():
        for model_name in models:
            model_data = mmlu_model_data[shot_type][model_name]
            
            # Only include models that have data
            if model_data['domain_count'] > 0:
                # Calculate averages across all domains
                avg_plain_acc = model_data['plain_acc_sum'] / model_data['domain_count']
                avg_kfold_acc = model_data['kfold_acc_sum'] / model_data['domain_count']
                avg_plain_rstd = model_data['plain_rstd_sum'] / model_data['domain_count']
                avg_kfold_rstd = model_data['kfold_rstd_sum'] / model_data['domain_count']
                
                # Calculate percentage changes
                acc_pct_change = ((avg_kfold_acc - avg_plain_acc) / avg_plain_acc * 100) if avg_plain_acc != 0 else np.nan
                rstd_pct_change = ((avg_kfold_rstd - avg_plain_rstd) / avg_plain_rstd * 100) if avg_plain_rstd != 0 else np.nan
                
                model_summaries.append({
                    'Family': family_name,
                    'Model': model_name,
                    'Avg Plain Acc': avg_plain_acc,
                    'Avg KFold Acc': avg_kfold_acc,
                    'Acc % Change': acc_pct_change,
                    'Avg Plain RSTD': avg_plain_rstd,
                    'Avg KFold RSTD': avg_kfold_rstd,
                    'RSTD % Change': rstd_pct_change,
                    'Domain Count': model_data['domain_count']
                })
    
    if model_summaries:
        df = pd.DataFrame(model_summaries)
        
        # Sort by Family and Model
        df = df.sort_values(['Family', 'Model'])
        
        # Format percentages and round decimals
        df['Acc % Change'] = df['Acc % Change'].map('{:+.2f}%'.format)
        df['RSTD % Change'] = df['RSTD % Change'].map('{:+.2f}%'.format)
        df['Avg Plain Acc'] = df['Avg Plain Acc'].map('{:.4f}'.format)
        df['Avg KFold Acc'] = df['Avg KFold Acc'].map('{:.4f}'.format)
        df['Avg Plain RSTD'] = df['Avg Plain RSTD'].map('{:.4f}'.format)
        df['Avg KFold RSTD'] = df['Avg KFold RSTD'].map('{:.4f}'.format)
        
        mmlu_individual_dfs[shot_type] = df
        
        # Save to CSV
        df.to_csv(f'mmlu_{shot_type}_model_summaries.csv', index=False)

# Display results
print("\n=== DATASET AND SHOT TYPE STATISTICS ===")
print(summary_df.to_string(index=False))

print("\nAnalysis complete. Results saved to dataset_shot_statistics.csv")

# Display MMLU individual model summaries
for shot_type in shot_types:
    if shot_type in mmlu_individual_dfs:
        print(f"\n=== MMLU MODEL SUMMARIES - {shot_type.upper()} ===")
        print(mmlu_individual_dfs[shot_type].to_string(index=False))
        print(f"\nResults saved to mmlu_{shot_type}_model_summaries.csv")

print("\n=== INTERPRETATION ===")
print("For overall dataset statistics:")
print("  - Accuracy % Difference: Positive values indicate KFold accuracy is higher than Plain accuracy")
print("  - Bias % Difference: Negative values indicate KFold bias is lower than Plain bias")
print("\nFor MMLU model summaries:")
print("  - Acc % Change: Positive values indicate KFold accuracy is higher than Plain accuracy")
print("  - RSTD % Change: Negative values indicate KFold RSTD (relative standard deviation) is lower than Plain RSTD")
print("\nA high positive Accuracy % Change and a high negative RSTD % Change would suggest")
print("that the KFold method improves model performance while reducing bias variation for MMLU.")
print("These metrics are averaged across all MMLU domains for each model.")