import re
import numpy as np

def is_maskable(tokenizer, token_id, id_to_maskable):
    """Return's True if token_id is maskable.
    """
    if token_id in id_to_maskable:
        return id_to_maskable[token_id]
    
    token_str = tokenizer.convert_ids_to_tokens(token_id)
        
    if token_str.startswith("Ġ"): # Ġ = Space prefix
        token_str = token_str[1:]
    
    id_to_maskable[token_id] = re.match(r"\b\w+\b", token_str) != None
    
    return id_to_maskable[token_id]

def get_maskable_positions(tokenizer, examples, id_to_maskable):
    """Get's the maskable positions in the given examples."""
    input_ids = examples["input_ids"]
    
    maskable_positions = []
    for seq in input_ids:
        maskable_positions.append([
            pos for pos, token_id in enumerate(seq) if is_maskable(tokenizer, token_id.item(), id_to_maskable)
        ])
    
    examples["maskable_positions"] = maskable_positions
    
    return examples

class MLMDataCollator:
    def __init__(self, tokenizer, mlm_prob = 0.15):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
    
    def collate(self, batch):
        """Randomly replace a portion of the input sequence tokens with the <mask> token."""
        input_ids = batch["input_ids"]
        maskable_positions = batch["maskable_positions"]
        
        labels = input_ids.clone()
        
        for i, maskable in enumerate(maskable_positions):
            selected_positions = np.random.choice(
                maskable, size = max(1, int(len(maskable) * self.mlm_prob))
            )
            
            input_ids[i][selected_positions] = self.tokenizer.mask_token_id
        
        return {"input_ids": input_ids, "labels": labels, "attention_mask": batch["attention_mask"]}