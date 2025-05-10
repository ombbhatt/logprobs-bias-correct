# Reducing Response Biases With Dataset-specific LogProbs Estimation

This repository contains the implementation of a Logprobs-based bias correction strategy that targets *yes-no* bias on binary-choice questions and *position-bias* on multiple-choice questions, and reduces biased behavior while maintaining or improving model accuracy and requiring zero overhead compute.

You can read more about the yes-no bias work in our CogSci 2025 paper [here]()

The position bias work is an extension of the same fundamental correction procedure to a more complicated question-format (and using a different bias metric), you can read the follow-up paper in the pre-print [here]()

## Installation and Set-up

After cloning the repo, simply create a `conda` environment and install the required packages from the `requirements.txt` file:

```
conda env create -n logprobsbias
pip install -r requirements.txt
```

## Datasets

We adapt five world-knowledge text datasets for testing our bias correction method and presenting results. Four datasets (COMPS, EWoK, bAbI, Arithmetic) are converted to "yes-no" format, and MMLU is used for position bias testing and essentially left unedited. 

***Note:*** Only EWoK and COMPS are used for the CogSci paper; the corection method is then further tested on the other yes-no datasets and extended to multiple-choice questions (i.e. MMLU) in the pre-print paper.

### Dataset Files

All datasets are available as zipped password-protected directories in the `datasets` folder. The password to access the four yes-no datasets (EWoK, COMPS, bAbI, Arith) is `yesno<dataset>pass1` (e.g. `yesnoewokpass1`). The password to access MMLU is `mcqmmlupass1`. We recommend using 7zip for zip extraction.

Scripts and CSV files other than the ones specified below can be ignored; they are not used for testing purposes and were intermediary assets in adapting the following datasets. All yes-no datasets are class-balanced.

1. **EWoK**: `ewokynq-data\` contains csv files with yes-no converted questions and answers, grouped by domain. The final yes-no dataset consists of 2056 context-question inputs across 11 domains. [Original dataset](https://ewok-core.github.io/).  

2. **COMPS**: `compsynq-data\comps_yn_rand_2100.csv` contains a yes-no converted subset of 2100 questions used in both papers for testing purposes. [Original dataset](https://github.com/kanishkamisra/comps). 

3. **bAbI**: `babiynq-data\` contains 18442 yes-no questions, collected and converted from 12 of the 20 domains of the original dataset, with no more than 2000 questions from any given domain. [Original dataset](https://github.com/facebookarchive/bAbI-tasks/tree/master/lua/babi)

4. **Arithmetic**: `arithynq-data\` contains 1200 yes-no questtions covering 1-digit and 2-digit addition, multiplication, and subtraction, derived from the Arithmetic subset of the BIGBench dataset. [Original dataset](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/arithmetic)

5. **MMLU**: `mmlu-data\` contains the original MMLU dataset in multiple-choice format, grouped by domain. The final dataset contains 14042 questions across 57 domains. [Original dataset](https://huggingface.co/datasets/cais/mmlu)

## Running Inference

**The main script to run LogProbs inference on the datasets is `unified_runner.py`**. The model, dataset, and other specifications are configured via command-line arguments, as explained below.

- `--dataset`: Can be `COMPS`, `EWOK`, `BABI`, `ARITH`, or `MMLU`
- `--model_family`: Can be `GPT2`, `Falcon`, `Qwen`, `Olmo`, or `Llama`. Specifying this will run *all* of the models specified within the family; model family lists can be edited within the script by deleting or adding names of other model versions available on HuggingFace.
- `--implementation`: Can be `yesnoplain`(Base inference for yes-no datasets), `yesnokfold` (i.e "Specific" method for bias correction as named in the CogSci paper), and `mcqplain` or `mcqkfold` following the same convention, but only when using the MMLU dataset. *Note: the code for "Generic" method for correction became defunct after implementation changes for the pre-print paper since it did not show promising de-biasing results; it will be added back in for reproducibility purposes soon.*
- `--batch_size`: Set according to compute constraints, default is 8
- `--domain`: Specify individual somains for processing for EWOK and MMLU. Default is 'all'. Other datasets do not have domain-wise grouping and argument need not be specified. Domain list can be found in `EWOK_DOMAINS` or `MMLU_DOMAINS` within the script.
- `--n_folds`: Defines the calibration-evaluation split ratio using 'folds' for the "Specific" correction method. Default is 5, i.e., data will be split into 5 equal folds (20% chunks), and thus 80% of the data will be used for calibration for processing (evaluating and correcting) each fold. Setting this to 4 will lead to four evaluation runs where 25% chunks of the data are evaluated and corrected using the remaining 75% as calibration data.
- `--prompt`: Can be `zeroshot` (question only), `fewshot` (Instruction + few-shot examples + question), or `instronly` (Instruction + question; this is an added prompt type only discussed in the pre-print paper).

**Example usage:**

Base inference:
```
python unified_runner.py --dataset EWOK --model_family Qwen --implementation yesnoplain --batch_size 8 --prompt zeroshot
```

Specific correction, domain and split-ratio specified:
```
python unified_runner.py --dataset EWOK --model_family Qwen --implementation yesnokfold --batch_size 8 --domain social_interactions --n_folds 5 --prompt fewshot
```

The other home directory scripts are core implementation files that are organized and called by `unified_runner.py`:
- `dataset_prompts.py`: Contains instruction-only and instruction + few-shot strings which are pre-fixed to dataset items during inference. 
- `balanced_folds_maker.py` : Logic for splitting dataset into balanced folds for implementing the "Specific" bias correction (which is the sole method discussed in the pre-print paper).
- `logprobs_yesno.py`: Question processing and bias correction logic for yes-no datasets.
- `logprobs_mcq.py`: Question processing and bias correction logic for multiple-choice (MMLU) dataset.
- `get_query_logprobs.py`: Base inference logprobs derivation function used by both yes-no and multiple-choice processing scripts mentioned above.

## Outputs

Outputs are CSV files ccontaining the question, ground-truth, model answer, log probabilities, and all other configuration relevant details. **These outputs are saved in the `outputs` directory. The latest and most complete set of results is found in `outputs/Mar-18-2025`. The `zeroshot`, `fewshot` and `instronly` directories contain detailed per-question model results per prompt-type. The password to access these files is `mar18<prompt-type>pass1`.** 

The result files in the zipped files are organized as `<prompt-type>/<dataset>/<implementation-type>/<model-family>/<domain>/<specific-model>_results.csv`. These are then used to generate the plots seen in the papers. The plots are in the `plots/` directory. 

*Note: The `<date>` designation for the outputs was simply a tracking mechanism for experiment runs; it can be specified near the top of the `unified_runner.py` script. The `<domain>` subdirectories are only organizationally useful for EWOK and MMLU; the other datasets simply have one domain subdirectory sharing the dataset's name.*

The `summary_jsons` contains useful aggregated model results for each model, prompt-type and dataset configuration, generated using the `extract_stats.py` script in `plotting-tables-scripts/`. 

The `mmlu_results` contains useful aggregated model performance data on MMLU, generated using the `acc-bias-mmlu.py` script in `plotting-tables-scripts/`. 

The `bias_analysis_detailed.csv` and `bias_analysis_summary.csv` contains bias comparison results between base and instruct-tuned counterparts of models, generated using the `instructvbase_stats.py` script in `plotting-tables-scripts/`.

The scripts to generate the scatterplots, heatmaps for yes-no datasets, and heatmap for MMLU are `scatterplots-new.py`, `heatmaps-new.py`, and `heatmap-mmlu.py` in `plotting-tables-scripts/`.
