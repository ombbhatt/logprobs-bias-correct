import torch, pandas as pd
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
from balanced_folds_maker import create_balanced_mcq_folds
from get_query_logprobs import get_query_logprobs
from dataset_prompts import get_dataset_prompts

def calculate_logprobs_batch(df, tokenizer, model, dataset=None, prompt=None):
    
    prompt_cond1, prompt_cond2 = get_dataset_prompts(dataset)

    results = []
        
    oa_variants = ["A", " A"]
    ob_variants = ["B", " B"]
    oc_variants = ["C", " C"]
    od_variants = ["D", " D"]

    all_variants = oa_variants + ob_variants + oc_variants + od_variants
    # print(f"all_variants: {all_variants}")
    
    # iterate through each row
    for i in range(len(df)):

        oa_logprobs = []
        ob_logprobs = []
        oc_logprobs = []
        od_logprobs = []
        
        row = df.iloc[i]
        question = row['question']
        option_a = row['A']
        option_b = row['B']
        option_c = row['C']
        option_d = row['D']

        og_query = f"{question}\nOptions: {option_a}, {option_b}, {option_c}, {option_d}"

        for variant in all_variants:
            if prompt == "fewshot":
                query = f"{prompt_cond1 + og_query}\nResponse:{variant}"
            elif prompt == "instronly":
                query = f"{prompt_cond2 + og_query}\nResponse:{variant}"
            elif prompt == "zeroshot":
                query = f"{og_query}\nResponse:{variant}"

            # print(query)

            input_ids = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
            logprob = get_query_logprobs(model, input_ids['input_ids'])[0]
            if variant in oa_variants:
                oa_logprobs.append(logprob)
            elif variant in ob_variants:
                ob_logprobs.append(logprob)
            elif variant in oc_variants:
                oc_logprobs.append(logprob)
            elif variant in od_variants:
                od_logprobs.append(logprob)
        
        oa_combined = torch.logsumexp(torch.tensor(oa_logprobs), dim=0).item()
        ob_combined = torch.logsumexp(torch.tensor(ob_logprobs), dim=0).item()
        oc_combined = torch.logsumexp(torch.tensor(oc_logprobs), dim=0).item()
        od_combined = torch.logsumexp(torch.tensor(od_logprobs), dim=0).item()
        
        results.append({
            'oa_logprob': oa_combined,
            'ob_logprob': ob_combined,
            'oc_logprob': oc_combined,
            'od_logprob': od_combined
        })
    
    return results

def process_dataset_mcq_combined(input_file, output_file, impl, model_name, model, tokenizer, domain, n_folds=5, batch_size=8, dataset=None, prompt=None):

    print("impl: ", impl)

    model_family = None
    dataset_folder = None

    if impl == "mcqkfold":
        print("impl is mcqkfold!!!")

        output_base1 = os.path.dirname(output_file) # this is just the domain folder path
        output_base2 = os.path.dirname(output_base1) # this is the parent folder of the domain folder i.e. the model family folder path
        model_family = os.path.basename(output_base2)
        output_base3 = os.path.dirname(output_base2) # this is the parent folder of the model family folder i.e. the implementation folder
        output_base4 = os.path.dirname(output_base3) # this is the parent folder of the implementation folder i.e. the dataset folder
        print(f"output_base4: {output_base4}")
        dataset_folder = output_base4
        
        mcqplain_output_file = os.path.join(output_base4, "mcqplain", model_family, domain, f"{model_name}_results.csv")
        if os.path.exists(mcqplain_output_file):
            print(f"Found mcqplain output file: {mcqplain_output_file}. Loading it to add kfold bias correction.")
            df = pd.read_csv(mcqplain_output_file)
            df = do_entire_kfold_thing(df, dataset, n_folds, domain)
            df.to_csv(output_file, index=False)
            return
        
        else:
            # break out and go to the else block below
            print(f"mcqplain output file: {mcqplain_output_file} does not exist. Proceeding with processing the dataset from scratch.")

    df = pd.read_csv(input_file, encoding='utf8')
    print(f"Starting processing domain: {domain} for model: {model_name}")
    
    # Initialize columns and metrics
    df['oa_logprob'] = None
    df['ob_logprob'] = None
    df['oc_logprob'] = None
    df['od_logprob'] = None

    df['predicted_answer'] = None
    df['is_correct'] = None
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i + batch_size]
        batch_results = calculate_logprobs_batch(
            batch_df, tokenizer, model, dataset, prompt
        )
        
        for j, result in enumerate(batch_results):
            if i + j >= len(df): break
            idx = i + j
            
            for key, value in result.items():
                df.loc[idx, key] = value

            logprob_cols = ['oa_logprob', 'ob_logprob', 'oc_logprob', 'od_logprob']
            logprobs = [float(df.loc[idx, col]) for col in logprob_cols]
            max_idx = np.argmax(logprobs) # index of the maximum logprob
            df.loc[idx, 'predicted_answer'] = ['A', 'B', 'C', 'D'][max_idx]
            df.loc[idx, 'is_correct'] = df.loc[idx, 'predicted_answer'] == df.loc[idx, 'answer']
        
        if i % (batch_size * 5) == 0:
            torch.cuda.empty_cache()

    # make plain_basepath directory if it doesn't exist
    if not os.path.exists(os.path.dirname(mcqplain_output_file)):
        plain_output_file = Path(dataset_folder) / f"mcqplain" / model_family / domain / f"{model_name}_results.csv"
        plain_output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(mcqplain_output_file, index=False)
    print("Plain Overall results:")
    print(f"Accuracy: {(df['is_correct'] == True).sum() / len(df):.3f}")
    print(f"Confusion matrix:")
    print(df.groupby(['answer', 'predicted_answer']).size())
    print(f"Recall rates:")
    print(df.groupby(['answer', 'predicted_answer']).size() / df.groupby('answer').size())

    if impl == "mcqkfold":
        df = do_entire_kfold_thing(df, dataset, n_folds, domain)
        df.to_csv(output_file, index=False)


def do_entire_kfold_thing(plain_df, dataset_name, n_folds, domain):

    total_questions = len(plain_df)
    fold_indices = None

    metrics = {'plain_recall_A': 0, 'plain_recall_B': 0, 'plain_recall_C': 0, 'plain_recall_D': 0, 'kfold_recall_A': 0, 'kfold_recall_B': 0, 'kfold_recall_C': 0, 'kfold_recall_D': 0}

    new_df = plain_df.copy()
    for col in ['eval_split_ratio', 'fold', 'raw_oa_logprob', 'raw_ob_logprob', 'raw_oc_logprob', 'raw_od_logprob', 'corrected_oa_logprob', 'corrected_ob_logprob', 'corrected_oc_logprob', 'corrected_od_logprob', 'calib_oa_mean', 'calib_ob_mean', 'calib_oc_mean', 'calib_od_mean', 'plain_predicted_answer', 'kfold_predicted_answer', 'plain_is_correct', 'kfold_is_correct']:
        new_df[col] = None

    new_df['eval_split_ratio'] = 1 / n_folds * 100 # tell us how much of the data is used for evaluation

    new_df['raw_oa_logprob'] = new_df['oa_logprob']
    new_df['raw_ob_logprob'] = new_df['ob_logprob']
    new_df['raw_oc_logprob'] = new_df['oc_logprob']
    new_df['raw_od_logprob'] = new_df['od_logprob']
    new_df.drop(columns=['oa_logprob', 'ob_logprob', 'oc_logprob', 'od_logprob'], inplace=True)

    # fold_indices = np.array_split(np.arange(total_questions), n_folds)
    df_reordered, fold_indices, balanced_size = create_balanced_mcq_folds(new_df, answer_column='answer', n_folds=5)
    new_df = df_reordered

    print(f"Domain: {domain}")

    # Verify fold contents
    # for i, indices in enumerate(fold_indices):
    #     fold_df = df_reordered.iloc[indices]
    #     if i < len(fold_indices) - 1:
    #         print(f"Fold {i+1} (balanced): {len(fold_df)} questions")
    #         print(fold_df['answer'].value_counts())
    #     else:
    #         print(f"Fold {i+1} (extra data): {len(fold_df)} questions")
    #         print(fold_df['answer'].value_counts())


    for fold_idx in range(n_folds):
        print(f"\nProcessing fold {fold_idx + 1}/{n_folds}")
        eval_indices = fold_indices[fold_idx]
        calib_indices = np.concatenate([fold_indices[i] for i in range(n_folds) if i != fold_idx and i != n_folds - 1]) # all other folds except the current eval fold and the extra data fold
        
        new_df.loc[eval_indices, 'fold'] = fold_idx

        # if the eval fold is the last fold, then it is the extra data fold
        # and the calib fold will be the entire balanced data
        if fold_idx == n_folds - 1:
            calib_indices = np.concatenate([fold_indices[i] for i in range(n_folds - 1)])

        # verify fold contents
        # print(f"Eval fold: {len(new_df.loc[eval_indices])} questions")
        # print(new_df.loc[eval_indices, 'answer'].value_counts())
        # print(f"Calib fold: {len(new_df.loc[calib_indices])} questions")
        # print(new_df.loc[calib_indices, 'answer'].value_counts())

        calib_oa_logprobs = new_df.loc[calib_indices, 'raw_oa_logprob'].values
        calib_ob_logprobs = new_df.loc[calib_indices, 'raw_ob_logprob'].values
        calib_oc_logprobs = new_df.loc[calib_indices, 'raw_oc_logprob'].values
        calib_od_logprobs = new_df.loc[calib_indices, 'raw_od_logprob'].values

        new_df.loc[eval_indices, 'calib_oa_mean'] = np.mean(calib_oa_logprobs)
        new_df.loc[eval_indices, 'calib_ob_mean'] = np.mean(calib_ob_logprobs)
        new_df.loc[eval_indices, 'calib_oc_mean'] = np.mean(calib_oc_logprobs)
        new_df.loc[eval_indices, 'calib_od_mean'] = np.mean(calib_od_logprobs)

        new_df.loc[eval_indices, 'corrected_oa_logprob'] = new_df.loc[eval_indices, 'raw_oa_logprob'] - np.mean(calib_oa_logprobs)
        new_df.loc[eval_indices, 'corrected_ob_logprob'] = new_df.loc[eval_indices, 'raw_ob_logprob'] - np.mean(calib_ob_logprobs)
        new_df.loc[eval_indices, 'corrected_oc_logprob'] = new_df.loc[eval_indices, 'raw_oc_logprob'] - np.mean(calib_oc_logprobs)
        new_df.loc[eval_indices, 'corrected_od_logprob'] = new_df.loc[eval_indices, 'raw_od_logprob'] - np.mean(calib_od_logprobs)

        highest_raw_logprob = new_df.loc[eval_indices, ['raw_oa_logprob', 'raw_ob_logprob', 'raw_oc_logprob', 'raw_od_logprob']].astype(float).idxmax(axis=1)

        highest_corrected_logprob = new_df.loc[eval_indices, ['corrected_oa_logprob', 'corrected_ob_logprob', 'corrected_oc_logprob', 'corrected_od_logprob']].astype(float).idxmax(axis=1)

        # Create mapping dictionaries
        raw_to_answer = {
            'raw_oa_logprob': 'A',
            'raw_ob_logprob': 'B',
            'raw_oc_logprob': 'C',
            'raw_od_logprob': 'D'
        }

        corrected_to_answer = {
            'corrected_oa_logprob': 'A',
            'corrected_ob_logprob': 'B',
            'corrected_oc_logprob': 'C',
            'corrected_od_logprob': 'D'
        }

        new_df.loc[eval_indices, 'plain_predicted_answer'] = highest_raw_logprob.map(raw_to_answer)
        new_df.loc[eval_indices, 'kfold_predicted_answer'] = highest_corrected_logprob.map(corrected_to_answer)

        correct_answers = new_df.loc[eval_indices, 'answer']

        new_df.loc[eval_indices, 'plain_is_correct'] = new_df.loc[eval_indices, 'plain_predicted_answer'].values == correct_answers.values
        new_df.loc[eval_indices, 'kfold_is_correct'] = new_df.loc[eval_indices, 'kfold_predicted_answer'].values == correct_answers.values

        for opts in ['A', 'B', 'C', 'D']:
            metrics[f'plain_recall_{opts}'] += sum((new_df.loc[eval_indices, 'plain_predicted_answer'] == opts) & (new_df.loc[eval_indices, 'answer'] == opts))
            metrics[f'kfold_recall_{opts}'] += sum((new_df.loc[eval_indices, 'kfold_predicted_answer'] == opts) & (new_df.loc[eval_indices, 'answer'] == opts)) 

    for opts in ['A', 'B', 'C', 'D']:
        total_opts = sum(new_df['answer'] == opts)
        metrics[f'plain_recall_{opts}'] /= total_opts
        metrics[f'kfold_recall_{opts}'] /= total_opts

    plain_std_of_recalls = np.std([metrics[f'plain_recall_{opts}'] for opts in ['A', 'B', 'C', 'D']])
    kfold_std_of_recalls = np.std([metrics[f'kfold_recall_{opts}'] for opts in ['A', 'B', 'C', 'D']])

    # plain accuracy based on 'plain_is_correct' column
    plain_accuracy = sum(new_df['plain_is_correct']) / total_questions

    # kfold accuracy based on 'kfold_is_correct' column
    kfold_accuracy = sum(new_df['kfold_is_correct']) / total_questions

    print(f"\nOverall Results:")
    print(f"Plain accuracy: {plain_accuracy:.3f}")
    print(f"Plain Recall: {metrics['plain_recall_A']}, {metrics['plain_recall_B']}, {metrics['plain_recall_C']}, {metrics['plain_recall_D']}")
    print(f"Plain RSTD: {plain_std_of_recalls:.3f}")
    print(f"Kfold accuracy: {kfold_accuracy:.3f}")
    print(f"Kfold Recall: {metrics['kfold_recall_A']}, {metrics['kfold_recall_B']}, {metrics['kfold_recall_C']}, {metrics['kfold_recall_D']}")
    print(f"Kfold RSTD: {kfold_std_of_recalls:.3f}")

    return new_df