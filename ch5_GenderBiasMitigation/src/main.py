import argparse
import yaml
import logging
import os
from pathlib import Path
import random
from transformers import BertForMaskedLM, BertTokenizer
import torch
import numpy as np

import train
import dataloader

logger = logging.getLogger(__name__)

# Setup logging
def setup_logging(log_file: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def set_seed(seed: int = 42):
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU + 1GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_cfg(cfg):
    cfg_dict = {}
    method_cfg = cfg['attention_guiding']
    layer_cfg = cfg['attention_layer_selection']
    head_cfg = cfg['attention_head_selection']
    
    # method
    if method_cfg['method'] not in method_cfg['allowed_methods']:
        logger.error(
            f"Invalid method: {method_cfg['method']}. "
            f"Must be one of {method_cfg['allowed_methods']}."
            )
        raise ValueError(
            f"Invalid method: {method_cfg['method']}. "
            f"Must be one of {method_cfg['allowed_methods']}."
            )
        
    cfg_dict['guiding_method'] = method_cfg['method']
    cfg_dict['guiding_direction'] = method_cfg['guiding_direction']
    
    # sample
    cfg_dict['num_train_sample'] = cfg['num_sample']
    
    # layer
    if layer_cfg['mode'] == 'fixed':
        cfg_dict['target_layers'] = layer_cfg['fixed_layers']
        
    elif layer_cfg['mode'] == 'random':
        random.seed(layer_cfg['seed'])  # シード固定
        selected_layers = random.sample(range(12), layer_cfg['num_layers'])  # 0~11から重複なしで選択
        cfg_dict['target_layers'] = selected_layers
    
    else:
        msg = "Please choose 'fixed' or 'random'. "
        logger.error(msg)
        raise ValueError(msg)
    
    # head
    if head_cfg['mode'] == 'fixed':
        cfg_dict['target_heads'] = head_cfg['fixed_heads']
        
    elif head_cfg['mode'] == 'random':
        random.seed(head_cfg['seed'])  # シード固定
        selected_heads = random.sample(range(12), head_cfg['num_heads'])  # 0~11から重複なしで選択
        cfg_dict['target_heads'] = selected_heads
    
    else:
        msg = "Please choose 'fixed' or 'random'. "
        logger.error(msg)
        raise ValueError(msg)
    
    logger.info(f'method: {cfg_dict["guiding_method"]}')
    logger.info(f'direction: {cfg_dict["guiding_direction"]}')
    logger.info(f'layers: {cfg_dict["target_layers"]}')
    logger.info(f'heads: {cfg_dict["target_heads"]}')
    logger.info(f'num_sample: {cfg_dict["num_train_sample"]}')
    
    return cfg_dict
        

def main():
    parser =  argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument('--train_data_file', default=None, type=str, required=True)
    parser.add_argument('--output_dir', default='results', type=str, required=True)
    
    ## Experiment configs
    parser.add_argument('--experiment_config', default=None, type=str, required=True, 
                        help="Path to YAML file that defines the experiment settings")
    
    ## Other parameters
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--model_name_or_path', default='bert-base-uncased', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--max_length', default=50, type=int)
    
    parser.add_argument('--train_batch_size', default=1, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--num_train_epochs', default=1, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('--no_cuda', action='store_true',
                        help='Avoid using CUDA when available')
    parser.add_argument('--seed', default=42, type=int)
    
    args = parser.parse_args()
    
    current_dir = Path(__file__).resolve().parent
    
    cfg_path = current_dir.parent / 'settings' / args.experiment_config
    cfg_name = cfg_path.stem
    method_name = cfg_path.parent.name
    
    logs_dir = current_dir.parent / "logs" / method_name 
    logs_dir.mkdir(parents=True, exist_ok=True)
    logs_file = logs_dir / f'{cfg_name}.log'
    setup_logging(logs_file)
    
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    cfg = load_cfg(cfg)
    print(cfg)
    
    set_seed(args.seed)
    
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertForMaskedLM.from_pretrained(args.model_name_or_path)
    
    # Dataset
    dataset_path = current_dir.parent / "dataset" / args.train_data_file
    args.train_data_file = dataset_path
    
    texts = dataloader.load_data(args, cfg)
    dataset = dataloader.MyDataset(cfg, texts, tokenizer)
    
    # Train
    output_dir = current_dir.parent / args.output_dir
    args.output_dir = output_dir / method_name / cfg_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Method: {method_name}")
    logger.info(f"Config: {cfg_name}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"seed: {args.seed}")

    model.to(args.device)
    train.train(args, cfg, dataset, model, tokenizer)
    
    #Evaluate
    
if __name__ == "__main__":
    main()