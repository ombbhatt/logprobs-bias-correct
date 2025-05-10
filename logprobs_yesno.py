import torch, pandas as pd
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
from balanced_folds_maker import create_balanced_folds
from get_query_logprobs import get_query_logprobs
from dataset_prompts import get_dataset_prompts

def calculate_logprobs_batch(contexts, questions, tokenizer, model, dataset=None, prompt=None):
    
    prompt_cond1, prompt_cond2 = get_dataset_prompts(dataset)

    results = []
        
    yes_variants = [" Yes", "Yes"]
    no_variants = [" No", "No"]
    all_variants = yes_variants + no_variants
    
    # Process each context-question pair
    for ctx, q in zip(contexts, questions):
        yes_logprobs = []
        no_logprobs = []
        
        # Process variants
        for variant in all_variants:
            if dataset == "COMPS" or dataset == "ARITH":
                query = ''.join(filter(None, [q.strip()]))
            elif dataset == "EWOK" or dataset == "BABI":
                query = ' '.join(filter(None, [ctx.strip(), q.strip()]))
            
            if prompt == "fewshot":
                query = f"{prompt_cond1 + query}\nResponse:{variant}"
            elif prompt == "instronly":
                query = f"{prompt_cond2 + query}\nResponse:{variant}"
            elif prompt == "zeroshot":
                query = query + variant

            # print(query)

            input_ids = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
            logprob = get_query_logprobs(model, input_ids['input_ids'])[0]
            if variant in yes_variants: 
                yes_logprobs.append(logprob)
            else:
                no_logprobs.append(logprob)
            del input_ids
        
        # Log-sum-exp for both yes and no variants
        yes_tensor = torch.tensor(yes_logprobs)
        no_tensor = torch.tensor(no_logprobs)
        
        yes_combined = torch.logsumexp(yes_tensor, dim=0).item()
        no_combined = torch.logsumexp(no_tensor, dim=0).item()
        
        results.append({
            'yes_logprob': yes_combined,
            'no_logprob': no_combined,
        })
    
    return results

def process_dataset_with_kfold_bias_correction(input_file, output_file, impl, model_name, model, tokenizer, domain, n_folds=5, batch_size=8, dataset=None, prompt=None):

    # if impl is "yesnokfold", first check if the output file for "yesnoplain" exists
    # if it does, we can just load the file and add the kfold bias correction to it
    print("impl: ", impl)

    model_family = None
    dataset_folder = None

    if impl == "yesnokfold":

        print("impl is yesnokfold!!!")

        # output file is defined as output_base / domain / f"{model_name}_results.csv"
        # and output_base is defined as "Path(f"outputs/{DATE}/{args.prompt}/{args.dataset}/{args.implementation}/{args.model_family}")"
        output_base1 = os.path.dirname(output_file) # this is just the domain folder path
        output_base2 = os.path.dirname(output_base1) # this is the parent folder of the domain folder i.e. the model family folder path
        model_family = os.path.basename(output_base2)
        output_base3 = os.path.dirname(output_base2) # this is the parent folder of the model family folder i.e. the implementation folder
        output_base4 = os.path.dirname(output_base3) # this is the parent folder of the implementation folder i.e. the dataset folder
        print(f"output_base4: {output_base4}")
        dataset_folder = output_base4
        
        yesnoplain_output_file = os.path.join(output_base4, "yesnoplain", model_family, domain, f"{model_name}_results.csv")
        if os.path.exists(yesnoplain_output_file):
            print(f"Found yesnoplain output file: {yesnoplain_output_file}. Loading it to add kfold bias correction.")
            df = pd.read_csv(yesnoplain_output_file)
            print("Plain Overall results:")
            print(f"Accuracy: {(df['is_correct'] == True).sum() / len(df):.3f}")
            print(f"Confusion matrix:")
            tp = ((df['predicted_answer'] == "Yes") & (df['Correct Answer'] == "Yes")).sum()
            tn = ((df['predicted_answer'] == "No") & (df['Correct Answer'] == "No")).sum()
            fp = ((df['predicted_answer'] == "Yes") & (df['Correct Answer'] == "No")).sum()
            fn = ((df['predicted_answer'] == "No") & (df['Correct Answer'] == "Yes")).sum()
            print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
            print(f"Bias score: {(tp + fp - tn - fn) / len(df):.3f}")

            df = do_entire_kfold_thing(df, dataset, n_folds)
            df.to_csv(output_file, index=False)
            return
        
        else:
            # break out and go to the else block below
            print(f"yesnoplain output file: {yesnoplain_output_file} does not exist. Proceeding with processing the dataset from scratch.")

        df = pd.read_csv(input_file)
        print(f"Starting processing domain: {domain} for model: {model_name}")
        
        # Initialize columns and metrics
        df['yes_logprob'] = None
        df['no_logprob'] = None

        df['predicted_answer'] = None
        df['is_correct'] = None
        
        for i in tqdm(range(0, len(df), batch_size)):
            batch_df = df.iloc[i:i + batch_size]
            batch_results = calculate_logprobs_batch(
                batch_df['Context'].tolist(),
                batch_df['Question'].tolist(),
                tokenizer, model, dataset, prompt
            )
            
            for j, result in enumerate(batch_results):
                if i + j >= len(df): break
                idx = i + j
                
                for key, value in result.items():
                    df.loc[idx, key] = value

                df.loc[idx, 'predicted_answer'] = "Yes" if df.loc[idx, 'yes_logprob'] > df.loc[idx, 'no_logprob'] else "No"
                df.loc[idx, 'is_correct'] = True if df.loc[idx, 'predicted_answer'] == df.loc[idx, 'Correct Answer'] else False
            
            if i % (batch_size * 5) == 0:
                torch.cuda.empty_cache()

        # make plain_basepath directory if it doesn't exist
        if not os.path.exists(os.path.dirname(yesnoplain_output_file)):
            plain_output_file = Path(dataset_folder) / f"yesnoplain" / model_family / domain / f"{model_name}_results.csv"
            plain_output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(yesnoplain_output_file, index=False)
        print("Plain Overall results:")
        print(f"Accuracy: {(df['is_correct'] == True).sum() / len(df):.3f}")
        print(f"Confusion matrix:")
        tp = ((df['predicted_answer'] == "Yes") & (df['Correct Answer'] == "Yes")).sum()
        tn = ((df['predicted_answer'] == "No") & (df['Correct Answer'] == "No")).sum()
        fp = ((df['predicted_answer'] == "Yes") & (df['Correct Answer'] == "No")).sum()
        fn = ((df['predicted_answer'] == "No") & (df['Correct Answer'] == "Yes")).sum()
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Bias score: {(tp + fp - tn - fn) / len(df):.3f}")

        if impl == "yesnokfold":
            df = do_entire_kfold_thing(df, dataset, n_folds)
            df.to_csv(output_file, index=False)


def do_entire_kfold_thing(plain_df, dataset_name, n_folds):

    total_questions = len(plain_df)
    fold_indices = None

    metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    new_df = plain_df.copy()
    for col in ['eval_split_ratio', 'fold', 'raw_yes_logprob', 'raw_no_logprob', 'corrected_yes_logprob', 
                'corrected_no_logprob', 'calib_yes_mean', 'calib_no_mean', 'raw_predicted_answer', 'raw_is_correct', 'kfold_predicted_answer', 'kfold_is_correct']:
        new_df[col] = None

    new_df['eval_split_ratio'] = 1 / n_folds * 100 # tell us how much of the data is used for evaluation

    new_df['raw_yes_logprob'] = new_df['yes_logprob']
    new_df['raw_no_logprob'] = new_df['no_logprob']
    new_df['raw_predicted_answer'] = new_df['predicted_answer']
    new_df['raw_is_correct'] = new_df['is_correct']
    new_df.drop(columns=['yes_logprob', 'no_logprob', 'predicted_answer', 'is_correct'], inplace=True)

    if dataset_name == "EWOK":
        # Split into groups of 4 questions
        group_starts = list(range(0, total_questions - 3, 4))
        # Split groups into folds
        fold_group_starts = np.array_split(group_starts, n_folds)
        fold_indices = [np.array([idx for start in fold_groups 
                        for idx in range(start, start + 4)])
                for fold_groups in fold_group_starts]
    
    elif dataset_name == "COMPS" or dataset_name == "BABI" or dataset_name == "ARITH":
        # Split into equal-sized folds
        # fold_indices = np.array_split(np.arange(total_questions), n_folds)
        fold_indices = create_balanced_folds(new_df, n_folds, answer_column='Correct Answer')

        for i, indices in enumerate(fold_indices):
            fold_df = new_df.iloc[indices]
            yes_count = (fold_df['Correct Answer'] == 'Yes').sum()
            no_count = (fold_df['Correct Answer'] == 'No').sum()
            print(f"Fold {i+1}: Yes={yes_count}, No={no_count}, Ratio={yes_count/no_count:.2f}")

    for fold_idx in range(n_folds):
        print(f"\nProcessing fold {fold_idx + 1}/{n_folds}")
        eval_indices = fold_indices[fold_idx]
        calib_indices = np.concatenate([fold_indices[i] for i in range(n_folds) if i != fold_idx])
        
        new_df.loc[eval_indices, 'fold'] = fold_idx

        calib_yes_logprobs = new_df.loc[calib_indices, 'raw_yes_logprob'].values
        calib_no_logprobs = new_df.loc[calib_indices, 'raw_no_logprob'].values

        new_df.loc[eval_indices, 'calib_yes_mean'] = np.mean(calib_yes_logprobs)
        new_df.loc[eval_indices, 'calib_no_mean'] = np.mean(calib_no_logprobs)

        # apply bias correction to eval set
        new_df.loc[eval_indices, 'corrected_yes_logprob'] = new_df.loc[eval_indices, 'raw_yes_logprob'] - np.mean(calib_yes_logprobs)
        new_df.loc[eval_indices, 'corrected_no_logprob'] = new_df.loc[eval_indices, 'raw_no_logprob'] - np.mean(calib_no_logprobs)

        # new_df.loc[eval_indices, 'bias_correction'] = bias_correction_term
        new_df.loc[eval_indices, 'kfold_predicted_answer'] = new_df.loc[eval_indices, 'corrected_yes_logprob'] > new_df.loc[eval_indices, 'corrected_no_logprob']
        new_df.loc[eval_indices, 'kfold_is_correct'] = new_df.loc[eval_indices, 'kfold_predicted_answer'] == (new_df.loc[eval_indices, 'Correct Answer'] == "Yes")

        metrics['TP'] += sum((new_df.loc[eval_indices, 'kfold_predicted_answer']) & (new_df.loc[eval_indices, 'Correct Answer'] == "Yes"))
        metrics['TN'] += sum(~(new_df.loc[eval_indices, 'kfold_predicted_answer']) & (new_df.loc[eval_indices, 'Correct Answer'] == "No"))
        metrics['FP'] += sum((new_df.loc[eval_indices, 'kfold_predicted_answer']) & (new_df.loc[eval_indices, 'Correct Answer'] == "No"))
        metrics['FN'] += sum(~(new_df.loc[eval_indices, 'kfold_predicted_answer']) & (new_df.loc[eval_indices, 'Correct Answer'] == "Yes"))

    # add summary row
    accuracy = (metrics['TP'] + metrics['TN']) / total_questions
    bias_score = (metrics['TP'] + metrics['FP'] - metrics['TN'] - metrics['FN']) / total_questions

    print(f"\nCorrected Overall Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Confusion Matrix: {metrics}")
    print(f"Bias score: {bias_score:.3f}")

    return new_df