import torch
from laylm.config import label as label_cfg
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


class FullMetrics(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, inputs, outputs):
        words, label_gts, label_preds =  normalized_words_labels_preds(
            inputs, outputs, 
            self.tokenizer
        )
        acc = accuracy_score(label_preds, label_gts)
        f1 = f1_score(label_preds, label_gts)
        precision = precision_score(label_preds, label_gts)
        recall = recall_score(label_preds, label_gts)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    
def normalized_words_labels_preds(inputs, outputs, tokenizer):
    preds = prediction_index(outputs)

    words, labels_gt, labels_pred = [],[],[]
    bsize = len(inputs['input_ids'])
    for idx in range(bsize):
        word_ids = inputs['input_ids'][idx].tolist()
        label_ids = inputs['labels'][idx].tolist()
        pred_ids = preds[idx].tolist()

        word, label_gt, label_pred = [],[],[]
        for (wds, lds, pds) in zip(word_ids, label_ids, pred_ids):
            if not (wds==tokenizer.cls_token_id or 
                    wds==tokenizer.sep_token_id or 
                    wds==tokenizer.pad_token_id):

                word.append(tokenizer.ids_to_tokens[wds])
                label_gt.append(label_cfg.idx_to_label[lds])
                label_pred.append(label_cfg.idx_to_label[pds])

        words.append(word)
        labels_gt.append(label_gt)
        labels_pred.append(label_pred)
        
    return words, labels_gt, labels_pred


def prediction_index(outputs):
    if len(outputs)>1:
        preds = outputs[1]
    else:
        preds = outputs[0]
    preds = torch.argmax(preds, dim=2)
    return preds