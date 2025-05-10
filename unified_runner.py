import argparse, os, gc, torch, sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, LlamaForCausalLM
from torch.cuda import empty_cache

gc.collect()
torch.cuda.empty_cache()

# Import implementations
from logprobs_yesno import process_dataset_with_kfold_bias_correction as process_dataset_kfold_response
from logprobs_mcq import process_dataset_mcq_combined as process_dataset_mcq_combined

DATE = "Mar-18-2025"

gpt2_models = ["gpt2"]

falcon_models = ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"]

mpt_models = ["mpt-7b", "mpt-7b-chat", "mpt-30b", "mpt-30b-chat"]

qwen_models = ["Qwen1.5-7B", "Qwen1.5-7B-Chat", "Qwen1.5-32B", "Qwen1.5-32B-Chat"]

olmo_models = ["Olmo-2-1124-7B", "Olmo-2-1124-7B-Instruct", "Olmo-2-1124-13B", "Olmo-2-1124-13B-Instruct"]

llama_models = ["Llama-2-7b-hf", "Llama-2-7b-chat-hf", "Llama-2-13b-hf", "Llama-2-13b-chat-hf"]

MODEL_CONFIGS = {
    "GPT2": {"models": gpt2_models, "model_class": GPT2LMHeadModel, "tokenizer_class": GPT2Tokenizer, "prefix": ""},

    "Falcon": {"models": falcon_models, "model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer, "prefix": "tiiuae/"},

    "MPT": {"models": mpt_models, "model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer, "prefix": "mosaicml/"},

    "Qwen": {"models": qwen_models, "model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer, "prefix": "Qwen/"},

    "Olmo": {"models": olmo_models, "model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer, "prefix": "allenai/"},

    "Llama": {"models": llama_models, "model_class": LlamaForCausalLM, "tokenizer_class": AutoTokenizer, "prefix": "meta-llama/"},
}

EWOK_DOMAINS = [
    "social_interactions", "social_properties", "material_dynamics", "social_relations", "quantitative_properties", "physical_dynamics", "physical_interactions", "material_properties", "physical_relations", "spatial_relations", "agent_properties"
    ]

COMPS_DOMAINS = ["comps"]

BABI_DOMAINS = ["babi"]

ARITH_DOMAINS = ["arith"]

MMLU_DOMAINS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology", "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science", "high_school_european_history", "high_school_geography", "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics", "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine", "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]

IMPLEMENTATIONS = {
    "yesnoplain": process_dataset_kfold_response,
    "yesnokfold": process_dataset_kfold_response,
    "mcqplain": process_dataset_mcq_combined,
    "mcqkfold": process_dataset_mcq_combined
}

def setup_model_and_tokenizer(model_name, model_family):
    config = MODEL_CONFIGS[model_family]
    full_model_name = f"{config['prefix']}{model_name}"
    is_large_model = any(x in model_name.lower() for x in ['7b', '9b', '13b', '2', "10b", "30b", "32b", "40b"])
    
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if is_large_model else None,
        "low_cpu_mem_usage": True
    }
    
    print(f"Loading model {full_model_name}...")
    model = config['model_class'].from_pretrained(
            full_model_name, **model_kwargs, 
            token=config.get('token'), 
            **({"use_mambapy" : False} if "Mamba" in full_model_name else {})
        )
            
    print("Loading tokenizer...")
    tokenizer = config['tokenizer_class'].from_pretrained(
        full_model_name,
        padding_side='left',
        truncation=True,
        token=config.get('token')
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    return model, tokenizer

def process_single_output(impl, model, tokenizer, model_name, implementation_fn, input_file, output_file, domain, n_folds=5, batch_size=4, dataset=None, prompt=None):
    if output_file.exists():
        print(f"Results exist for {model_name}, skipping...")
        return
        
    kwargs = {
        # impl is the key for the implementation_fn from the IMPLEMENTATIONS dict
        "impl": impl,
        "model_name": model_name,
        "model": model,
        "tokenizer": tokenizer,
        "domain": domain,
        **({"n_folds": n_folds} if "kfold" in str(implementation_fn) else {}),
        "batch_size": batch_size,
        "dataset": dataset,
        "prompt": prompt
    }
    
    implementation_fn(str(input_file), str(output_file), **kwargs)
    if hasattr(model, 'clear_kv_cache'):
        model.clear_kv_cache()
    elif hasattr(model, 'reset_kv_cache'):
        model.reset_kv_cache()

def check_all_outputs_exist(model_name, domains, output_base):
    """Check if all required output files exist for a model across all domains"""
    for domain in domains:
            output_file = output_base / domain / f"{model_name}_results.csv"
            if not output_file.exists():
                return False
    return True

def process_model_across_domains(impl, model_name, model_family, implementation_fn, domains, dataset_path, output_base, n_folds=5, batch_size=4, dataset=None, prompt=None):
    """Process a single model across all domains"""

    # Check if all outputs exist before loading model
    if check_all_outputs_exist(model_name, domains, output_base):
        print(f"All results exist for {model_name}, skipping...")
        return
    
    # Else Load model once
    model, tokenizer = setup_model_and_tokenizer(model_name, model_family)
    
    try:
        for domain in domains:
            print(f"\nProcessing domain: {domain}")
            print(f"\nProcessing model: {model_name}")
            
            # Setup input file based on dataset
            if dataset == "EWOK":
                input_file = dataset_path / f"processed_t2q_{domain}.csv"
            elif dataset == "COMPS":
                input_file = dataset_path / f"comps_yn_rand_2prop_2100.csv"
            elif dataset == "BABI":
                input_file = dataset_path / f"babi-ynq-big.csv"
            elif dataset == "ARITH":
                input_file = dataset_path / f"arith-ynq-big.csv"
            elif dataset == "MMLU":
                input_file = dataset_path / f"{domain}_processed.csv"
            
            # if not input_file.exists():
            #     continue
            
            output_file = output_base / domain / f"{model_name}_results.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            process_single_output(
                impl, model, tokenizer, model_name, implementation_fn,
                input_file, output_file, domain, n_folds,
                batch_size, dataset, prompt
            )
    
    finally:
        # Clean up model after all domains are processed
        del model, tokenizer
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        except RuntimeError:
            pass
        gc.collect()

def main():
    
    parser = argparse.ArgumentParser(description="Run model inference across different models and domains")
    parser.add_argument("--dataset", type=str, choices=["EWOK", "COMPS", "BABI", "ARITH", "MMLU"], required=True)
    parser.add_argument("--model_family", type=str, choices=list(MODEL_CONFIGS.keys()), required=True)
    parser.add_argument("--implementation", type=str, choices=list(IMPLEMENTATIONS.keys()), required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--domain", type=str, choices=EWOK_DOMAINS + ["all"], default="all")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--prompt", choices=["fewshot", "zeroshot", "instronly"], default="fewshot")
    args = parser.parse_args()
    
    if not torch.cuda.is_available() and input("CUDA unavailable. Continue? (y/n): ").lower() != 'y':
        sys.exit(0)
    
    # Setup paths
    if args.dataset == "EWOK":
        dataset_path = Path("ewokynq-scripts-data/t2q_nodup_nominpairs")
    elif args.dataset == "COMPS":
        dataset_path = Path("compsynq-scripts-data")
    elif args.dataset == "BABI":
        dataset_path = Path("babiynq-scripts-data")
    elif args.dataset == "ARITH":
        dataset_path = Path("arithynq-scripts-data")
    elif args.dataset == "MMLU":
        dataset_path = Path("mmlu-scripts-data")
    elif not dataset_path.exists():
        print(f"Dataset path {dataset_path} does not exist. Exiting...")
        sys.exit(1)
    
    output_base = Path(f"outputs/{DATE}/{args.prompt}/{args.dataset}/{args.implementation}/{args.model_family}")
    if args.dataset == "EWOK":
        domains = (EWOK_DOMAINS if args.domain == "all" else [args.domain])
    elif args.dataset == "COMPS":
        domains = COMPS_DOMAINS
    elif args.dataset == "BABI":
        domains = BABI_DOMAINS
    elif args.dataset == "ARITH":
        domains = ARITH_DOMAINS
    elif args.dataset == "MMLU":
        domains = MMLU_DOMAINS
    
    implementation_fn = IMPLEMENTATIONS[args.implementation]
    impl = args.implementation
    
    try:
        for model_name in MODEL_CONFIGS[args.model_family]["models"]:
            print(f"\nProcessing model: {model_name}")
            process_model_across_domains(
                impl,
                model_name=model_name,
                model_family=args.model_family,
                implementation_fn=implementation_fn,
                domains=domains,
                dataset_path=dataset_path,
                output_base=output_base,
                n_folds=args.n_folds,
                batch_size=args.batch_size,
                dataset=args.dataset,
                prompt=args.prompt
            )
                    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        empty_cache()
        gc.collect()
        sys.exit(1)

if __name__ == "__main__":
    main()