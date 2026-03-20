from transformers import BertTokenizer, BertForMaskedLM
import torch

class MaskPredicter:
    def __init__(self, tokenizer, mlm_model, dataset, device=None):
        self.tokenizer = tokenizer
        self.mlm_model = mlm_model
        self.dataset = dataset
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.mlm_model.to(self.device)
        
        self.he_ids = self.tokenizer.convert_tokens_to_ids("he")
        self.she_ids = self.tokenizer.convert_tokens_to_ids("she")
        
    def mask_sentence(self, index):
        """
        性別単語のみが異なる文章ペアから[MASK]文を作成
        
        """
        # male text
        male_text = self.dataset[index]['male']
        male_encoding = self.tokenizer(male_text, return_tensors="pt")
        male_input_ids = male_encoding["input_ids"][0]  # バッチ0番目のID列
        male_tokens = self.tokenizer.convert_ids_to_tokens(male_input_ids)
        
        #female text
        female_text = self.dataset[index]['female']
        female_encoding = self.tokenizer(female_text, return_tensors="pt")
        female_input_ids = female_encoding["input_ids"][0]
        female_tokens = self.tokenizer.convert_ids_to_tokens(female_input_ids)
        
        # mask index
        gender_indices = [i for i, (x, y) in enumerate(zip(male_tokens, female_tokens)) if x != y]
        
        masked_input_ids = male_encoding["input_ids"].clone()
        for idx in gender_indices:
            masked_input_ids[0][idx] = self.tokenizer.mask_token_id

        masked_inputs = {
            "input_ids": masked_input_ids,
            "attention_mask": male_encoding["attention_mask"]
        }
        if "token_type_ids" in male_encoding:
            masked_inputs["token_type_ids"] = male_encoding["token_type_ids"]
            
        masked_tokens = self.tokenizer.convert_ids_to_tokens(masked_inputs["input_ids"][0])
        masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)

        return masked_inputs, {
            "occupation" : self.dataset[index]['occupation'],
            "male" : male_text,
            "female" : female_text,
            "masked_index" : gender_indices,
            "masked_text" : masked_text
            }
        
    def predict_mask(self, masked_inputs):
        masked_inputs = {k: v.to(self.device) for k, v in masked_inputs.items()}

        self.mlm_model.eval()
        with torch.no_grad():
            outputs = self.mlm_model(**masked_inputs, output_attentions=True)
            logits = outputs.logits
            #attentions = outputs.attentions
            
        # [MASK]の位置を取得
        mask_token_index = (masked_inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        # [MASK]位置の確率を計算
        probs = torch.softmax(logits[0, mask_token_index[0]], dim=-1)  # shape: (num_mask, vocab_size)
            
        he_prob = probs[self.he_ids].item()
        she_prob = probs[self.she_ids].item()
        
        return {
            'he_prob' : he_prob,
            'she_prob' : she_prob
        }
            
            
        