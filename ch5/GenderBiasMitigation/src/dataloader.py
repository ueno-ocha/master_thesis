import logging
logger = logging.getLogger(__name__)

from torch.utils.data import Dataset
import torch
import target_attention_generator
import json


def pad_mat(mat, max_len):
    L = mat.size(0)
    out = torch.zeros(max_len, max_len)
    out[:L, :L] = mat
    return out

def make_attn_mask(L, max_len):
    mask = torch.zeros(max_len, max_len)
    mask[:L, :L] = 1
    return mask

class MyDataset(Dataset):
    def __init__(self, cfg, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        self.cfg = cfg
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        item = {}
        
        inputs = self.tokenizer(self.texts[idx]['main'])
        item['inputs'] = inputs
        
        #行列固定型
        if self.cfg['guiding_method'] in ['profession_to_gender']:
            attn_matrix = target_attention_generator.occupation_to_gender_matrix(inputs) #Attention行列を作る
            item['syntax_attn'] = attn_matrix
        
        #行列固定型
        elif self.cfg['guiding_method'] in ['tokens_to_gender']:
            attn_matrix = target_attention_generator.context_to_gender_matrix(inputs) #Attention行列を作る
            item['syntax_attn'] = attn_matrix
        
        #行列作成型
        elif self.cfg['guiding_method'] in ['gender_attention_averaging']:
            sub_inputs = self.tokenizer(self.texts[idx]['sub'])
            item['sub_inputs'] = sub_inputs
        
        return item
    
class MyCollator:
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.pad_id = tokenizer.pad_token_id
        
    def __call__(self, features):
        if self.cfg['guiding_method'] in ['profession_to_gender', 'tokens_to_gender']:
            max_len = max(len(f["inputs"]["input_ids"]) for f in features)
            
            input_ids = []
            attention_mask = []

            for f in features:
                L = len(f["inputs"]["input_ids"])
                pad_len = max_len - L

                input_ids.append(
                    f["inputs"]["input_ids"] + [self.pad_id] * pad_len
                )
                attention_mask.append(
                    f["inputs"]["attention_mask"] + [0] * pad_len
                )
            
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            
            syntax_attn = torch.stack([
                pad_mat(f["syntax_attn"], max_len)
                for f in features])
            
            syntax_attn_mask = torch.stack([
                make_attn_mask(len(f["inputs"]["input_ids"]), max_len)
                for f in features
                ])
            
            return {
                "inputs": {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                },
                "syntax_attn": syntax_attn,
                "syntax_attn_mask": syntax_attn_mask,
            }
        
        elif self.cfg['guiding_method'] in ['gender_attention_averaging']:
            max_len = max(len(f["inputs"]["input_ids"]) for f in features)
            
            input_ids = []
            attention_mask = []

            for f in features:
                L = len(f["inputs"]["input_ids"])
                pad_len = max_len - L

                input_ids.append(
                    f["inputs"]["input_ids"] + [self.pad_id] * pad_len
                )
                attention_mask.append(
                    f["inputs"]["attention_mask"] + [0] * pad_len
                )
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
                
            sub_input_ids = []
            sub_attention_mask = []

            for f in features:
                L = len(f["sub_inputs"]["input_ids"])
                pad_len = max_len - L

                sub_input_ids.append(
                    f["sub_inputs"]["input_ids"] + [self.pad_id] * pad_len
                )
                sub_attention_mask.append(
                    f["sub_inputs"]["attention_mask"] + [0] * pad_len
                )
            sub_input_ids = torch.tensor(sub_input_ids)
            sub_attention_mask = torch.tensor(sub_attention_mask)
            
            syntax_attn_mask = torch.stack([
                make_attn_mask(len(f["inputs"]["input_ids"]), max_len)
                for f in features
                ])
            
            return {
                "inputs": {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    },
                "sub_inputs": {
                    "input_ids": sub_input_ids,
                    "attention_mask": sub_attention_mask,
                    },
                "syntax_attn_mask": syntax_attn_mask
                }

def load_data(args, cfg):
    # Dataset
    with open(args.train_data_file, 'r') as f:
        data = json.load(f)
    texts = []
        
    num = cfg['num_train_sample'] // 2
    male_biased = data[:num] 
    female_biased = data[-num:]
    
    if cfg['guiding_direction'] == 'stereotype':
        for item in male_biased:
            texts.append({'main': item['texts']['male'], 'sub': item['texts']['female']})
            
        for item in female_biased:
            texts.append({'main': item['texts']['female'], 'sub': item['texts']['male']})
    
    elif cfg['guiding_direction'] == 'anti-stereotype':
        for item in male_biased:
            texts.append({'main': item['texts']['female'], 'sub': item['texts']['male']})
        
        for item in female_biased:
            texts.append({'main': item['texts']['male'], 'sub': item['texts']['female']})
    
    else:
        for item in male_biased:
            texts.append({'main': item['texts']['male'], 'sub': item['texts']['female']})
        
        for item in female_biased:
            texts.append({'main': item['texts']['male'], 'sub': item['texts']['female']})
    
    logger.info(texts)
    return texts