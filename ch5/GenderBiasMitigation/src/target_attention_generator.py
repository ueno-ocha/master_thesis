import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#職業単語の定義
OCCUPATION_TOKEN_IDS = []
OCCUPATION_WORDS = ['carpenter', 'driver', 'sheriff', 'developer', 'farmer', 'guard', 'chief', 'lawyer',
                    'cook', 'physician', 'ceo', 'analyst', 'manager', 'supervisor', 'editor', 'designer',
                    'accountant', 'auditor', 'writer', 'baker', 'clerk', 'counselor', 'attendant', 'teacher',
                    'sewer', 'librarian', 'assistant', 'cleaner', 'housekeeper', 'nurse', 'receptionist', 'secretary']
for w in OCCUPATION_WORDS:
    token_id = tokenizer.convert_tokens_to_ids(w)
    OCCUPATION_TOKEN_IDS.append(token_id)

#print(OCCUPATION_TOKEN_IDS)


# 職業 → 性別(主語)のAttentionを強める
def occupation_to_gender_matrix(inputs):
    input_ids = inputs['input_ids']
    seq_len = len(input_ids)

    pos = None
    for i in range(seq_len):
        if input_ids[i] in OCCUPATION_TOKEN_IDS:
            pos = i
            break

    if pos is None:
        raise ValueError(
            'Text does not contain any occupation words.'
        )

    attn = torch.zeros(seq_len, seq_len)
    GENDER_POS = 1     # Assumes "[CLS] She is a ..." template
    attn[pos, GENDER_POS] = 1.0

    return attn


# 文章中の単語 → 性別(主語)のAttentionを強める
def context_to_gender_matrix(inputs):
    input_ids = inputs['input_ids']
    seq_len = len(input_ids)
    
    attn = torch.zeros(seq_len, seq_len)
    GENDER_POS = 1     # Assumes "[CLS] She is a ..." template
    attn[:, GENDER_POS] = 1.0

    return attn