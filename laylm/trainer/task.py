import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

from .metrics import FullMetrics

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers


class TaskLayoutLM(pl.LightningModule):
    def __init__(self, model, tokenizer, grad_clip=1.0, hparams={}):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.grad_clip = grad_clip
        self.hparams = hparams
        self.metrics = FullMetrics(tokenizer)
        
    def forward(
        self,  
        input_ids,
        bbox,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None
    ):
        return self.model(
            input_ids,
            bbox,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            labels
        )
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
    def shared_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch[0].to(self.device),
            "attention_mask": batch[1].to(self.device),
            "token_type_ids": batch[2].to(self.device),
            "labels": batch[3].to(self.device),
            "bbox": batch[4].to(self.device)
        }
        
        outputs = self.model(**inputs)
        loss, logits = outputs[0], outputs[1]
        
        metrics = self.metrics(inputs, outputs)
        
        return loss, logits, metrics
        
        
    def training_step(self, batch, batch_idx):
        loss, logits, metrics = self.shared_step(batch, batch_idx)
        self.log('trn_loss', loss, prog_bar=True, logger=True)
        self.log('trn_acc', metrics['accuracy'], prog_bar=True, logger=True)
        self.log('trn_f1', metrics['f1'], prog_bar=True, logger=True)
        self.log('trn_precision', metrics['precision'], prog_bar=True, logger=True)
        self.log('trn_recall', metrics['recall'], prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logits, metrics = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', metrics['accuracy'], prog_bar=True, logger=True)
        self.log('val_f1', metrics['f1'], prog_bar=True, logger=True)
        self.log('val_precision', metrics['precision'], prog_bar=True, logger=True)
        self.log('val_recall', metrics['recall'], prog_bar=True, logger=True)
        return loss
        
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW( optimizer_grouped_parameters, lr=5e-5, eps=1e-8 )
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1)
        
        return [optimizer], [scheduler]
    