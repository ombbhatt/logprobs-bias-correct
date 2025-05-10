import torch

def get_query_logprobs(model, query_input_ids):
        with torch.no_grad():
            device = next(model.parameters()).device
            outputs = model(query_input_ids.to(device))
            log_probs = torch.log_softmax(outputs.logits[:, -2, :], dim=-1) # Get logprobs for second-to-last token (just before the appended answer token)
            # logits dimension: (batch_size, sequence_length, vocab_size)
            # so ':' means all batch_size, '-2' means second-to-last token, ':' means all vocab_size
            
            try:
                if torch.cuda.is_available() and torch.cuda.is_initialized() and "Qwen" not in str(model) and "Falcon" not in str(model):
                    torch.cuda.empty_cache()
            except RuntimeError:
                pass
            
            return [log_probs[0, query_input_ids[0, -1]].item()]