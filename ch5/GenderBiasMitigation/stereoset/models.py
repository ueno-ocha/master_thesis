# This code is adapted from:
# https://github.com/McGill-NLP/bias-bench
# Only BERT-related components are extracted and simplified.

from functools import partial

import torch
import transformers

class BertModel:
    def __new__(self, model_name_or_path):
        return transformers.BertModel.from_pretrained(model_name_or_path)
    
class BertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.BertForMaskedLM.from_pretrained(model_name_or_path)


