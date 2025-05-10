import os
import pandas as pd
import argparse
from collections import defaultdict
import json
import numpy as np

def extract_acc_bias_yes_no(file_path):
    """Extract accuracy and bias for yes/no datasets."""
    plain_counts = {'YesAnswer': 0, 'NoAnswer': 0, 'Correct': 0, 'Incorrect': 0}
    kfold_counts = {'YesAnswer': 0, 'NoAnswer': 0, 'Correct': 0, 'Incorrect': 0}
    df = pd.read_csv(file_path)
    # go through each line 
    for index, row in df.iterrows():
        if row['raw_predicted_answer'] == 'Yes':
            plain_counts['YesAnswer'] += 1
        elif row['raw_predicted_answer'] == 'No':
            plain_counts['NoAnswer'] += 1
        if row['raw_predicted_answer'] == row['Correct Answer']:
            plain_counts['Correct'] += 1
        else:
            plain_counts['Incorrect'] += 1

        if row['kfold_predicted_answer'] == True:
            kfold_counts['YesAnswer'] += 1
        elif row['kfold_predicted_answer'] == False:
            kfold_counts['NoAnswer'] += 1
        if (row['kfold_predicted_answer'] == True and row['Correct Answer'] == "Yes") or (row['kfold_predicted_answer'] == False and row['Correct Answer'] == "No"):
            kfold_counts['Correct'] += 1
        else:
            kfold_counts['Incorrect'] += 1

    # calculate acc and bias score
    plain_acc = plain_counts['Correct'] / (plain_counts['Correct'] + plain_counts['Incorrect'])
    kfold_acc = kfold_counts['Correct'] / (kfold_counts['Correct'] + kfold_counts['Incorrect'])
    plain_bias = (plain_counts['YesAnswer'] - plain_counts['NoAnswer']) / (plain_counts['YesAnswer'] + plain_counts['NoAnswer'])
    kfold_bias = (kfold_counts['YesAnswer'] - kfold_counts['NoAnswer']) / (kfold_counts['YesAnswer'] + kfold_counts['NoAnswer'])

    return plain_bias, kfold_bias, plain_acc, kfold_acc

def extract_acc_recall_mcq(file_path):
    """Extract accuracy and recall statistics for multiple choice questions."""
    options = ['A', 'B', 'C', 'D']
    
    # Initialize counters
    plain_counts = {'Correct': 0, 'Total': 0}
    kfold_counts = {'Correct': 0, 'Total': 0}
    
    # For each option, initialize true positives and actual positives
    plain_option_tp = {option: 0 for option in options}
    plain_option_actual = {option: 0 for option in options}
    kfold_option_tp = {option: 0 for option in options}
    kfold_option_actual = {option: 0 for option in options}
    
    df = pd.read_csv(file_path)
    
    for index, row in df.iterrows():
        # Count total
        plain_counts['Total'] += 1
        kfold_counts['Total'] += 1
        
        # Get true answer and predicted answers
        true_answer = row['answer']
        plain_predicted = row['plain_predicted_answer']
        kfold_predicted = row['kfold_predicted_answer']
        
        # Count correct predictions
        if plain_predicted == true_answer:
            plain_counts['Correct'] += 1
        if kfold_predicted == true_answer:
            kfold_counts['Correct'] += 1
        
        # Update actual counts for each option
        plain_option_actual[true_answer] += 1
        kfold_option_actual[true_answer] += 1
        
        # Update true positives for each option
        if plain_predicted == true_answer:
            plain_option_tp[true_answer] += 1
        if kfold_predicted == true_answer:
            kfold_option_tp[true_answer] += 1
    
    # Calculate accuracies
    plain_acc = plain_counts['Correct'] / plain_counts['Total'] if plain_counts['Total'] > 0 else 0
    kfold_acc = kfold_counts['Correct'] / kfold_counts['Total'] if kfold_counts['Total'] > 0 else 0
    
    # Calculate recalls for each option
    plain_recalls = {}
    kfold_recalls = {}
    
    for option in options:
        if plain_option_actual[option] > 0:
            plain_recalls[option] = plain_option_tp[option] / plain_option_actual[option]
        else:
            plain_recalls[option] = 0
            
        if kfold_option_actual[option] > 0:
            kfold_recalls[option] = kfold_option_tp[option] / kfold_option_actual[option]
        else:
            kfold_recalls[option] = 0
    
    # Calculate recall standard deviations
    plain_recall_values = list(plain_recalls.values())
    kfold_recall_values = list(kfold_recalls.values())
    
    plain_rstd = np.std(plain_recall_values) if plain_recall_values else 0
    kfold_rstd = np.std(kfold_recall_values) if kfold_recall_values else 0
    
    return plain_acc, kfold_acc, plain_recalls, kfold_recalls, plain_rstd, kfold_rstd

def get_latest_date_dir(base_dir):
    """Get the most recent date directory."""
    date_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not date_dirs:
        return None
    # Sort by date (assuming format is consistent)
    return sorted(date_dirs)[-1]

def collect_stats(base_dir, prompt_format, model_family, dataset, specific_date=None):
    """Collect statistics for the specified parameters."""
    
    # Define constants based on your file structure
    DATASET_DOMAINS = {
        "EWOK": ["social_interactions", "social_properties", "material_dynamics", "social_relations", 
                 "quantitative_properties", "physical_dynamics", "agent_properties", "physical_interactions", 
                 "material_properties", "physical_relations", "spatial_relations"],
        "COMPS": ["comps"],
        "BABI": ["babi"],
        "ARITH": ["arith"],
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
    
    ALL_MODEL_FAMILIES = {
        'Falcon': ["Falcon3-10B-Base", "Falcon3-10B-Instruct", "Falcon3-3B-Base", "Falcon3-3B-Instruct"],
        'MPT': ["mpt-7b", "mpt-7b-chat", "mpt-30b", "mpt-30b-chat"],
        'Qwen': ["Qwen1.5-7B", "Qwen1.5-7B-Chat", "Qwen1.5-32B", "Qwen1.5-32B-Chat"],
        'Llama': ["Llama-2-7b-hf", "Llama-2-7b-chat-hf", "Llama-2-13b-hf", "Llama-2-13b-chat-hf"]
    }
    
    # Validate inputs
    if prompt_format not in ["zeroshot", "fewshot", "instronly"]:
        raise ValueError(f"Invalid prompt format: {prompt_format}")
    
    if model_family not in ALL_MODEL_FAMILIES:
        raise ValueError(f"Invalid model family: {model_family}")
    
    if dataset not in DATASET_DOMAINS:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    # Determine which date directory to use
    if specific_date:
        date_dir = specific_date
        if not os.path.isdir(os.path.join(base_dir, date_dir)):
            raise ValueError(f"Date directory not found: {date_dir}")
    else:
        date_dir = get_latest_date_dir(base_dir)
        if not date_dir:
            raise ValueError("No date directories found")
    
    # Determine the correct subfolder based on dataset type
    if dataset == "MMLU":
        subfolder = "mcqkfold"
    else:
        subfolder = "yesnokfold"
    
    # Build the path to the dataset directory
    dataset_path = os.path.join(base_dir, date_dir, prompt_format, dataset, subfolder, model_family)
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Path not found: {dataset_path}")
    
    # Dictionary to store results
    results = defaultdict(lambda: defaultdict(dict))
    
    # Get the relevant domains for this dataset
    domains = DATASET_DOMAINS[dataset]
    
    # Get the models for this model family
    models = ALL_MODEL_FAMILIES[model_family]
    
    # Collect stats for each domain and model
    for domain in domains:
        domain_path = os.path.join(dataset_path, domain)
        if not os.path.isdir(domain_path):
            print(f"Warning: Domain directory not found: {domain_path}")
            continue
        
        for model in models:
            csv_file = f"{model}_results.csv"
            file_path = os.path.join(domain_path, csv_file)
            
            if not os.path.isfile(file_path):
                print(f"Warning: Results file not found: {file_path}")
                continue
            
            try:
                if dataset == "MMLU":
                    # Use MCQ-specific extraction function
                    plain_acc, kfold_acc, plain_recalls, kfold_recalls, plain_rstd, kfold_rstd = extract_acc_recall_mcq(file_path)
                    
                    results[domain][model] = {
                        "plain_acc": plain_acc,
                        "kfold_acc": kfold_acc,
                        "plain_recalls": plain_recalls,
                        "kfold_recalls": kfold_recalls,
                        "plain_rstd": plain_rstd,
                        "kfold_rstd": kfold_rstd
                    }
                    
                    print(f"Processed: {domain} - {model}")
                    print(f"  Plain Accuracy: {plain_acc:.4f}")
                    print(f"  K-fold Accuracy: {kfold_acc:.4f}")
                    print(f"  Plain Recalls: {', '.join([f'{k}: {v:.4f}' for k, v in plain_recalls.items()])}")
                    print(f"  K-fold Recalls: {', '.join([f'{k}: {v:.4f}' for k, v in kfold_recalls.items()])}")
                    print(f"  Plain Recall Std Dev: {plain_rstd:.4f}")
                    print(f"  K-fold Recall Std Dev: {kfold_rstd:.4f}")
                    print()
                else:
                    # Use Yes/No-specific extraction function
                    plain_bias, kfold_bias, plain_acc, kfold_acc = extract_acc_bias_yes_no(file_path)
                    
                    results[domain][model] = {
                        "plain_bias": plain_bias,
                        "kfold_bias": kfold_bias,
                        "plain_acc": plain_acc,
                        "kfold_acc": kfold_acc
                    }
                    
                    print(f"Processed: {domain} - {model}")
                    print(f"  Plain Bias: {plain_bias:.4f}")
                    print(f"  K-fold Bias: {kfold_bias:.4f}")
                    print(f"  Plain Accuracy: {plain_acc:.4f}")
                    print(f"  K-fold Accuracy: {kfold_acc:.4f}")
                    print()
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Extract accuracy and bias statistics from CSV files")
    parser.add_argument("--prompt_format", required=True, choices=["zeroshot", "fewshot", "instronly"], 
                        help="Prompt format to analyze")
    parser.add_argument("--model_family", required=True, choices=["Falcon", "MPT", "Qwen", "Llama"], 
                        help="Model family to analyze")
    parser.add_argument("--dataset", required=True, choices=["EWOK", "COMPS", "BABI", "ARITH", "MMLU"], 
                        help="Dataset to analyze")
    parser.add_argument("--date", required=True, help="Specific date directory to use (e.g., 'Mar-18-2025')")
    parser.add_argument("--output", help="Path to save the results JSON file")
    
    args = parser.parse_args()
    
    # Adjust the base directory to look up one level from the stats-scripts folder
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    
    results = collect_stats(
        base_dir, 
        args.prompt_format, 
        args.model_family, 
        args.dataset, 
        specific_date=args.date
    )
    
    # Print summary
    print("\nSummary:")
    for domain, models in results.items():
        print(f"\nDomain: {domain}")
        for model, stats in models.items():
            print(f"  Model: {model}")
            for stat_name, stat_value in stats.items():
                if stat_name in ['plain_recalls', 'kfold_recalls']:
                    # For recall dictionaries, print each option separately
                    print(f"    {stat_name}:")
                    for option, recall in stat_value.items():
                        print(f"      {option}: {recall:.4f}")
                else:
                    print(f"    {stat_name}: {stat_value:.4f}")
    
    # Save results to file if specified
    if args.output:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()