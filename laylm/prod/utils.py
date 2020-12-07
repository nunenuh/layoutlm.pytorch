import torch


import pandas as pd

from laylm.trainer import metrics
from laylm.config import label as label_cfg
from laylm.config import token as token_cfg




def annoset_inputs(data_dict, device):
    input_ids = torch.tensor(data_dict['token_ids'], dtype=torch.long)
    mask = torch.tensor(data_dict['mask'], dtype=torch.long)
    bbox = torch.tensor(data_dict['bboxes'], dtype=torch.long)
    
    input_data = {
        'input_ids': input_ids.unsqueeze(dim=0).to(device),
        'attention_mask': mask.unsqueeze(dim=0).to(device),
        'bbox': bbox.unsqueeze(dim=0).to(device)
    }
    return input_data


def annoset_transform(objects, tokenizer, max_seq_length = 512):
    data_anno = tokenize_duplicate_dict(objects, tokenizer)
    texts, bboxes, tokens, token_ids, wseq, gseq, mask = [],[],[],[],[],[],[]

    texts.append(token_cfg.cls_token)
    bboxes.append(token_cfg.cls_token_box)
    tokens.append(token_cfg.cls_token)
    token_ids.append(token_cfg.cls_token_id)
    wseq.append(token_cfg.ignore_index_token_id)
    gseq.append(token_cfg.ignore_index_token_id)
    mask.append(1)

    for obj in data_anno:
        texts.append(obj['text'])
        bboxes.append(obj['bbox'])
        tokens.append(obj['token'])
        token_ids.append(obj['token_id'])
        wseq.append(obj['wseq'])
        gseq.append(obj['gseq'])
        mask.append(1)
        

    texts.append(token_cfg.sep_token)
    bboxes.append(token_cfg.sep_token_box)
    tokens.append(token_cfg.sep_token)
    token_ids.append(token_cfg.sep_token_id)
    wseq.append(token_cfg.ignore_index_token_id)
    gseq.append(token_cfg.ignore_index_token_id)
    mask.append(1)
    

    
    pad_length = max_seq_length - len(texts)
    for p in range(pad_length):
        texts.append(token_cfg.pad_token)
        bboxes.append(token_cfg.pad_token_box)
        tokens.append(token_cfg.pad_token)
        token_ids.append(token_cfg.pad_token_id)
        wseq.append(token_cfg.ignore_index_token_id)
        gseq.append(token_cfg.ignore_index_token_id)
        mask.append(0)
    
    data_dict = {
        'words':texts,
        'bboxes': bboxes,
        'tokens': tokens,
        'token_ids': token_ids,
        'mask': mask,
        'gseq': gseq,
        'wseq': wseq
    }
    
    return data_dict


def tokenize_duplicate_dict(objects, tokenizer):
    new_objects = []
    gseq = 0
    for idx, obj in enumerate(objects):
        curr_text = objects[idx]['text']
        token = tokenizer.tokenize(curr_text)
        if len(token) > 1:
            wseq = 0
            for tok in token:
                
                new_obj = objects[idx].copy()
                new_obj['token'] = tok
                new_obj['token_id'] = tokenizer.convert_tokens_to_ids(tok)
                new_obj['fraction'] = True
                new_obj['wseq'] = wseq
                new_obj['gseq'] = gseq
                new_objects.append(new_obj)
                wseq+=1
                
            gseq+=1
                
        else:
            if len(token)==0:
                obj['token'] = '[UNK]'
                obj['token_id'] = tokenizer.convert_tokens_to_ids('[UNK]')
            else:
                obj['token'] = token[0]
                obj['token_id'] = tokenizer.convert_tokens_to_ids(token[0])
                
            
            obj['fraction'] = False
            obj['wseq'] = 0
            obj['gseq'] = gseq
            new_objects.append(obj)
            gseq+=1

    return new_objects


def normalized_prediction(outputs, tokenizer):
    preds = prediction_index(outputs)
    
    bsize = preds.shape[0]
    
    labels = []
    for idx in range(bsize):
        label_pred = []
        for pds in preds[idx].tolist():
            lbl = label_cfg.idx_to_label.get(pds, "O")
            label_pred.append(lbl)
        labels.append(label_pred)
    
    return labels

    
def prediction_index(outputs):
    if len(outputs)>1:
        preds = outputs[1]
    else:
        preds = outputs[0]
    preds = torch.argmax(preds, dim=2)
    return preds

def clean_prediction_data(data_dict, tokenizer):
    words = data_dict['words']
    boxes = data_dict['bboxes']
    tokens = data_dict['tokens']
    labels = data_dict['labels']
    gseq = data_dict['gseq']
    wseq = data_dict['wseq']

    data = {
        'words':[],
        'bboxes': [],
        'tokens': [],
        'labels': [],
        'gseq': [],
        'wseq': [],
    }

    for (w,b,t,l,gq,wq) in zip(words, boxes, tokens, labels, gseq, wseq):
        if not (w==tokenizer.cls_token or 
                w==tokenizer.sep_token or 
                w==tokenizer.pad_token):

            data['words'].append(w)
            data['bboxes'].append(b)
            data['tokens'].append(t)
            data['labels'].append(l)
            data['gseq'].append(gq)
            data['wseq'].append(wq)
            
    return data

def sort_multidim(data):
    sorter = lambda x: (x[2][1], x[1])
    # x[2][1] sort by y position
    # x[1] sort by BILOU
    
    return sorted(data, key=sorter)


def word_taken(data):
    str_out = ""
    for idx in range(len(data)):
        w = data[idx][0]
        if w!="" and len(w)!=0:
            str_out += w
            if idx!=len(data)-1:
                str_out += " "
            
    return str_out



def rebuild_prediction_data(data):
    df = pd.DataFrame(data)
    dfg = df.groupby('gseq').aggregate({
        'words': 'min', 
        'bboxes':'last',
        'tokens':'sum',
        'labels':'first'
    })
    
    base_data = dict((k,[]) for k,v in label_cfg.base_label_name.items())
    for idx in range(len(dfg)):
        labels = dfg.iloc[idx]['labels']
        bbox = dfg.iloc[idx]['bboxes']
        if not labels=="O":
            bil, val = labels.split("-")
            val_type, val_label = val.split("_")
            if val_type=="VAL":
                word = dfg.iloc[idx]['words']
                key = label_cfg.label_to_name[val_label]
                base_data[key].append((word, bil, bbox))


    for k,v in base_data.items():
        sorted_data = sort_multidim(v)
        base_data[k] = word_taken(sorted_data)
    
    return base_data
    