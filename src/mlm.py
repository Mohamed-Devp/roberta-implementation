import re
import numpy as np

def is_maskable(tokenizer, token_id, id_to_maskable):
    """Return's True if token_id is maskable."""    
    if token_id in id_to_maskable:
        return id_to_maskable[token_id]
    
    token_str = tokenizer.convert_ids_to_tokens(token_id)
        
    if token_str.startswith("Ġ"): # Ġ = Space prefix
        token_str = token_str[1:]
    
    id_to_maskable[token_id] = re.match(r"\b\w+\b", token_str) != None
    
    return id_to_maskable[token_id]

def get_maskable_ids(tokenizer, examples, id_to_maskable):
    """Return's the maskable indices in the examples input_ids."""
    maskable_ids = []
    
    for sequence in examples["input_ids"]:
        maskable_seq = []
        
        for index, token_id in enumerate(sequence):
            if is_maskable(tokenizer, token_id.item(), id_to_maskable):
                maskable_seq.append(index)
        
        maskable_ids.append(maskable_seq)
    
    return maskable_ids

def apply_masking(tokenizer, examples, mlm_prob):
    """Randomly replace a portion of the input sequence tokens with the <mask> token."""
    input_ids = examples["input_ids"]
    labels = input_ids.clone()
    
    maskbale_ids = examples["maskable_ids"]
    
    for i, maskable in enumerate(maskbale_ids):
        selected_positions = np.random.choice(
                maskable, size = max(1, int(len(maskable) * mlm_prob))
        )

        input_ids[i, selected_positions] = tokenizer.mask_token_id
    
    return {
        "input_ids": input_ids,
        "attention_mask": examples["attention_mask"],
        "labels": labels
    }