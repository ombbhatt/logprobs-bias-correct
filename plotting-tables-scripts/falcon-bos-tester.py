from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model 
model_name = "tiiuae/Falcon3-10B-Base" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# Function to get token from ID
def get_token_from_id(token_id):
    # Convert token ID back to string
    token = tokenizer.decode([token_id])
    return token

# Example: Try with a token ID
token_id = 1  # you can change this to any valid token ID
print(f"Token ID {token_id} corresponds to: '{get_token_from_id(token_id)}'")

# You can also see some special tokens and their IDs
print("\nSpecial tokens:")
print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id if tokenizer.pad_token else 'None'})")
print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id if tokenizer.bos_token else 'None'})")