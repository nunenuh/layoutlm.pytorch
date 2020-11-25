from collections import OrderedDict
from ..config import label as label_cfg
from ..config import token as token_cfg
from ..ops import boxes_ops
import numpy as np
import json


def format_annotation_objects(anno, tokenizer, max_seq_length=512):
    objects = anno['objects']
    objects = prepare_objects_data(objects, tokenizer)
    
    tokens, labels, label_ids, boxes = [],[],[],[]
    
    tokens.append(token_cfg.cls_token)
    boxes.append(token_cfg.cls_token_box)
    label_ids.append(token_cfg.pad_token_label_id)
    
    for obj in objects:
        tokens.append(obj['token'])
        labels.append(obj['label'])

        lab = label_cfg.label_to_idx[obj['label']]
        label_ids.append(lab)

        pts = obj['points']
        pts = np.array(pts)
        pts = boxes_ops.order_points(np.array(pts))
        pts = list(boxes_ops.to_xyminmax(pts))
        boxes.append(pts)

    tokens.append(token_cfg.sep_token)
    boxes.append(token_cfg.sep_token_box)
    label_ids.append(token_cfg.pad_token_label_id)

    input_ids = tokenizer.encode(tokens, add_special_tokens=False)
    input_masks = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    
    padding_data_ouput = padding_data(
        input_ids, input_masks, 
        segment_ids, label_ids, boxes,
        max_seq_length=max_seq_length
    )
    input_ids, input_masks, segment_ids, label_ids, boxes = padding_data_ouput
    
    return input_ids, input_masks, segment_ids, label_ids, boxes


def padding_data(input_ids, input_masks, segment_ids,
                label_ids, boxes, max_seq_length=512, 
                pad_on_left=False):
    
    padding_length = max_seq_length - len(input_ids)

    if not pad_on_left:
        input_ids += [token_cfg.pad_token] * padding_length
        input_masks += [0] * padding_length
        segment_ids += [0] * padding_length
        label_ids += [token_cfg.pad_token_label_id] * padding_length
        boxes += [token_cfg.pad_token_box] * padding_length
    else:
        input_ids = [token_cfg.pad_token] * padding_length + input_ids
        input_masks = [0] * padding_length + input_masks
        segment_ids = [0] * padding_length + segment_ids
        label_ids = [token_cfg.pad_token_label_id] * padding_length + label_ids
        boxes = [token_cfg.pad_token_box] * padding_length + boxes
        
    return input_ids, input_masks, segment_ids, label_ids, boxes


def prepare_objects_data(objects, tokenizer):
    duplicated_objects = tokenize_duplicate_dict(objects, tokenizer)
    formatted_objects = reformat_label_oriented(duplicated_objects)
    bilou_objects = inject_bilou_to_objects(formatted_objects)
    objects = revert_to_list_format(bilou_objects)
    
    return objects


def tokenize_duplicate_dict(objects, tokenizer):
    new_objects = []
    for idx, obj in enumerate(objects):
        curr_text = objects[idx]['text']

        token = tokenizer.tokenize(curr_text)
        if len(token) > 1:
            for tok in token:
                new_obj = objects[idx].copy()
                new_obj['token'] = tok
                new_objects.append(new_obj)
        else:
            if len(token)==0:
                obj['token'] = ''
            else:
                obj['token'] = token[0]
            new_objects.append(obj)

    return new_objects


def reformat_label_oriented(objects):
    data = OrderedDict({k:{'field':[], 'delimiter':[], 'value':[]} for k,v in label_cfg.base_label_name.items()})

    for idx, obj in enumerate(objects):
        cname_curr = obj['classname']
        scname_curr = obj['subclass']
        data[cname_curr][scname_curr].append(obj)

    return data


def inject_bilou_to_objects(objects):
    for idx, (key,val) in enumerate(objects.items()):
        field = val['field']
        delim = val['delimiter']
        value = val['value']

        if len(field)>0:
            objects[key]['field'] = inject_bilou_to_label(field)

        if len(delim)>0:
            objects[key]['delimiter'] = inject_bilou_to_label(delim)

        if len(value)>0:
            objects[key]['value'] = inject_bilou_to_label(value)
    
    return objects


def inject_bilou_to_label(data_dict):
    # create bilou prefix to dictionary data
    texts = []
    for idx in range(len(data_dict)):
        texts.append(data_dict[idx]['token'])
    bil_prefix = bilou_prefixer(texts)

    #inject bilou prefix into label inside data_dict
    for idx, (bil, fld) in enumerate(zip(bil_prefix, data_dict)):
        if fld['label'] != "O":
            label = bil+'-'+fld['label']
            data_dict[idx]['label'] = label
    
    return data_dict


def revert_to_list_format(dnew):
    data_list = []
    for k,v in dnew.items():
        field = dnew[k]['field']
        delim = dnew[k]['delimiter']
        value = dnew[k]['value']
        if len(delim)>0:
            line_list = field+delim+value
        else:
            line_list = field+value

        data_list += line_list
    return data_list


# datas['objects']
def bilou_prefixer(text_list, label=None):
    out = []
    text_len = len(text_list)
    if text_len==1:
        bl = "U"
        if label!=None: bl =  bl + "-" + label
        out.append(bl)
    elif text_len>1:
        for idx, text in enumerate(text_list):
            if idx==0: 
                bl = "B"
                if label!=None: bl = bl + "-" + label
                out.append(bl)
            elif idx < text_len - 1: 
                bl = "I"
                if label!=None: bl = bl + "-" + label
                out.append(bl)
            else: 
                bl = "L"
                if label!=None: bl =  bl + "-" + label
                out.append(bl)
    return out


def tokenize_inside_dict(data_dict, tokenizer):
    for idx in range(len(data_dict)):
        text = data_dict[idx]['text']
        data_dict[idx]['text'] = tokenizer.tokenize(text)
    return data_dict
