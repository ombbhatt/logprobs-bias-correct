import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_balanced_folds(df, n_folds, answer_column):
        # Get the answers
        y = df[answer_column].values  
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Generate the indices for each fold
        fold_indices = []
        for _, fold_idx in skf.split(np.zeros(len(df)), y):
            fold_indices.append(fold_idx)
        
        return fold_indices


def create_balanced_mcq_folds(df, answer_column='answer', n_folds=5, random_state=42):
    # Count occurrences of each answer option
    answer_counts = df[answer_column].value_counts()
    min_count = answer_counts.min()
    
    # We need to make sure our balanced dataset size is:
    # 1. Divisible by 4 (for equal A/B/C/D)
    # 2. Divisible by n_folds-1 (for equal fold sizes)
    samples_per_answer = (min_count // (n_folds-1)) * (n_folds-1)
    
    balanced_indices = []
    np.random.seed(random_state)
    
    for answer in ['A', 'B', 'C', 'D']:
        option_indices = df[df[answer_column] == answer].index.tolist()
        # Take exactly samples_per_answer samples from each answer type
        sampled_indices = np.random.choice(option_indices, samples_per_answer, replace=False)
        balanced_indices.extend(sampled_indices)
    
    # get "extra" indices (those not in the balanced subset)
    all_indices = set(df.index)
    extra_indices = list(all_indices - set(balanced_indices))
    
    # new order for dataframe: balanced data first, then extra data
    new_order = balanced_indices + extra_indices
    
    reordered_df = df.loc[new_order].reset_index(drop=True)
    
    # number of balanced samples
    balanced_data_size = len(balanced_indices)
    
    # create balanced folds
    fold_indices = []
    samples_per_fold = balanced_data_size // (n_folds-1)
    samples_per_answer_per_fold = samples_per_fold // 4
    
    # make n_folds-1 balanced folds
    for fold_idx in range(n_folds-1):
        fold_indices_list = []
        for answer_idx, answer in enumerate(['A', 'B', 'C', 'D']):
            start_idx = answer_idx * samples_per_answer
            fold_start = start_idx + fold_idx * samples_per_answer_per_fold
            fold_end = fold_start + samples_per_answer_per_fold
            fold_indices_list.extend(range(fold_start, fold_end))
        fold_indices.append(np.array(fold_indices_list))
    
    # Add the extra data as the last fold
    extra_fold = np.arange(balanced_data_size, len(reordered_df))
    fold_indices.append(extra_fold)
    
    return reordered_df, fold_indices, balanced_data_size