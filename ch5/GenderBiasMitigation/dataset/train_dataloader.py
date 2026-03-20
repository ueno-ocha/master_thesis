import argparse
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from gender_dataloader import CustomDataset
from mask_predictor import MaskPredicter
from transformers import BertTokenizer, BertForMaskedLM

#職業単語の定義
OCCUPATION_WORDS = ['carpenter', 'driver', 'sheriff', 'developer', 'farmer', 'guard', 'chief', 'lawyer',
                    'cook', 'physician', 'CEO', 'analyst', 'manager', 'supervisor', 'editor', 'designer',
                    'accountant', 'auditor', 'writer', 'baker', 'clerk', 'counselor', 'attendant', 'teacher',
                    'sewer', 'librarian', 'assistant', 'cleaner', 'housekeeper', 'nurse', 'receptionist', 'secretary']
        

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--male_data_path', default=None, type=str, required=True,
                        help='The input data file (including male word)')
    parser.add_argument('--female_data_path', default=None, type=str, required=True,
                        help='The input data file (including female word)')
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help='The output directory where the model predictions.')
    
    args = parser.parse_args()
    
    with open(args.male_data_path, 'r') as f:
        male_data = json.load(f)
    filtered_male_data = {k: v for k, v in male_data.items() if k in OCCUPATION_WORDS}
    print(len(filtered_male_data.keys()))
    
    with open(args.female_data_path, 'r') as f:
        female_data = json.load(f)
    filtered_female_data = {k: v for k, v in female_data.items() if k in OCCUPATION_WORDS}
    print(len(filtered_male_data.keys()))
    
    dataset = CustomDataset(filtered_male_data, filtered_female_data)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    
    maskprediction = MaskPredicter(tokenizer, model, dataset)
    
    data_list = []
    for i in tqdm(range(len(dataset))):
        masked_inputs, texts = maskprediction.mask_sentence(i)
        probs = maskprediction.predict_mask(masked_inputs)
        
        mlm_bias = probs['he_prob'] - probs['she_prob']
        
        data_dict = {}
        data_dict['mlm_bias'] = mlm_bias
        data_dict['texts'] = texts
        
        data_list.append(data_dict)
    
    data_list = sorted(data_list, key=lambda x: x['mlm_bias'], reverse=True)
    
    output_path = os.path.join(args.output_dir, 'result_mlm_new.json')
    
    with open(output_path, 'w') as f:
        json.dump(data_list, f, indent=4)

if __name__ == "__main__":
    main()

        
        
        
        
    