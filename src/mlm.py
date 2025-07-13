import re
import numpy as np
import torch

def is_maskable(tokenizer, token_id, id_to_maskable):
    """Return's True if token_id is maskable."""    
    if token_id in id_to_maskable:
        return id_to_maskable[token_id]
    
    token_str = tokenizer.convert_ids_to_tokens(token_id)
        
    if token_str.startswith("Ġ"): # Ġ = Space prefix
        token_str = token_str[1:]
    
    id_to_maskable[token_id] = re.match(r"\b\w+\b", token_str) != None
    
    return id_to_maskable[token_id]

def get_maskable(tokenizer, examples, id_to_maskable):
    """Return's a boolean mask indicating maskable positions in input_ids."""
    maskable = []
    
    for sequence in examples["input_ids"]:
        maskable.append([
            is_maskable(tokenizer, token_id, id_to_maskable) for token_id in sequence
        ])
    
    return maskable

def apply_masking(tokenizer, examples, mlm_prob):
    """Randomly replace a portion of the input sequence tokens with the <mask> token."""
    input_ids = examples["input_ids"]
    attention_mask = examples["attention_mask"]
    labels = input_ids.clone()
    
    maskable = examples["maskable"]
    for i, mask in enumerate(maskable):
        indices = torch.where(mask)[0]
        
        selected = np.random.choice(
            indices, size = max(1, int(len(indices) * mlm_prob))
        )
        
        input_ids[i, selected] = tokenizer.mask_token_id
    
    return {
        "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels
    }