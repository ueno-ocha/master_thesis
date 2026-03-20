import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataloader

class AttentionGuidingLoss:
    def __init__(self, cfg):
        self.layers = cfg['target_layers']
        self.heads = cfg['target_heads']
        self.criterion = nn.MSELoss(reduction="none")

    def __call__(self, pred_attn, teacher, attn_mask):
        total = 0

        for layer in self.layers:
            pred = pred_attn[layer][:, self.heads]
            tgt  = teacher.get(layer, self.heads)

            mask = attn_mask.unsqueeze(1).expand_as(pred)
            mse = self.criterion(pred, tgt) * mask
            total += mse.sum() / mask.sum()

        return total / len(self.layers)
                
                 
class TeacherAttention:
    def __init__(self, mode, attn, static_attn=None, sub_attn=None):
        self.mode = mode
        self.attn = attn
        self.static_attn = static_attn      # [B, L, L]
        self.sub_attn = sub_attn            # tuple([B,H,L,L])
    
    def get(self, layer, heads):
        """
        returns: [B, |heads|, L, L]
        """
        if self.mode == "static":
            attn = self.static_attn.unsqueeze(1)
            return attn.expand(-1, len(heads), -1, -1)

        elif self.mode == "averaging":
            main = self.attn[layer][:, heads]
            sub = self.sub_attn[layer][:, heads]
            
            mean_attn = (main + sub) / 2
            
            return mean_attn
            #attn_list = [self.sub_attn[layer][:, heads] for layer in layers]  # 各 layer ごとの [B, |heads|, L, L]
            #return attn_list
            # 同じ layer & 同じ head をそのまま使う
            #return self.sub_attn[layers][:, heads]            
                

def train(args, cfg, train_dataset, model, tokenizer):
    
    collator = dataloader.MyCollator(cfg, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collator)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Lerning rate = %e", args.learning_rate)
    logger.info("  Train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    
    global_step = 0
    tr_loss = 0.0        #学習ループ全体でのloss
    model.zero_grad()
    
    for epoch in range(args.num_train_epochs):
        model.train()
        train_loss = 0.0         # 1　epoch内での累積loss
        tr_num = 0               # 累積したバッチの数
        
        bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, batch in bar:
            inputs = {
                k: v.to(args.device)
                for k, v in batch["inputs"].items()
                }

            # forward
            outputs = model(**inputs, output_attentions=True)
            
            if cfg['guiding_method'] in ['profession_to_gender', 'tokens_to_gender']:
                teacher = TeacherAttention(
                    mode="static",
                    attn=outputs.attentions,
                    static_attn=batch["syntax_attn"].to(args.device)
                    )
            
            elif cfg['guiding_method'] in ['gender_attention_averaging']:
                sub_inputs = {
                    k: v.to(args.device)
                    for k, v in batch["sub_inputs"].items()
                    }
                with torch.no_grad():
                    sub_outputs = model(**sub_inputs, output_attentions=True)
                
                teacher = TeacherAttention(
                    mode="averaging",
                    attn=outputs.attentions,
                    sub_attn=sub_outputs.attentions
                    )

            # custom loss
            loss_func = AttentionGuidingLoss(cfg)
            loss = loss_func(
                pred_attn=outputs.attentions,
                teacher=teacher,
                attn_mask=batch["syntax_attn_mask"].to(args.device)
                )

            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            # 勾配計算
            loss.backward()

            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                #scheduler.step()
                optimizer.zero_grad()  
                global_step += 1
                #output_flag = True
                
            # tqdmで平均 loss 表示
            train_loss += loss.item()
            tr_num += 1
            avg_loss = train_loss / tr_num
            bar.set_description(f"epoch {epoch} avg_loss {avg_loss:.5f}")
            
    # 学習終了後に最終モデル保存
    save_path = args.output_dir
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
        
    print(f"Model saved to {save_path}")
    logging.info(f"Model saved to {save_path}")    
        
    return global_step, train_loss / tr_num   