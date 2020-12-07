import torch
import torch.nn as nn
import torch.optim as optim


import torch
import torch.nn as nn
import torch.optim as optim
from . import utils as utils

from laylm.config import label as label_cfg
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    LayoutLMConfig,
    LayoutLMForTokenClassification,
    get_linear_schedule_with_warmup,
)


# tokenizer = BertTokenizer.from_pretrained(
#     "indobenchmark/indobert-base-p2",
#     do_lower_case=True,
#     cache_dir=None,
# )

# config = LayoutLMConfig.from_pretrained(
#     "microsoft/layoutlm-base-uncased",
#     num_labels=label_cfg.num_labels,
#     cache_dir=None
# )

# model = LayoutLMForTokenClassification.from_pretrained(
#     'microsoft/layoutlm-base-uncased',
#     config=config,
# #     return_dict=True
# )

# model.resize_token_embeddings(len(tokenizer))

class Extractor(object):
    def __init__(self, tokenizer=None, weight=None, device='cpu'):
        self.device = device
        self.tokenizer = tokenizer
        self._load_tokenizer()
        self._load_config()
        self._load_model()
        
        self.weight = weight
        if self.weight != None:
            self.load_state_dict(weight)
            
    
        
        
    def load_state_dict(self, state_dict_path):
        state_dict = torch.load(state_dict_path)
        self.model.load_state_dict(state_dict)
    
    def _load_tokenizer(self):
        if self.tokenizer == None:
            self.tokenizer = BertTokenizer.from_pretrained(
            "indobenchmark/indobert-base-p2",
            do_lower_case=True,
            cache_dir=None,
        )
            
    def _load_config(self):
        self.config = LayoutLMConfig.from_pretrained(
            "microsoft/layoutlm-base-uncased",
            num_labels=label_cfg.num_labels,
            cache_dir=None
        )
    
    def _load_model(self):
        self.model = LayoutLMForTokenClassification.from_pretrained(
                'microsoft/layoutlm-base-uncased',
                 config=self.config,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        
    
    def predict(self, objects):
        data_dict = utils.annoset_transform(objects, self.tokenizer, max_seq_length=512)
        inputs_data = utils.annoset_inputs(data_dict, device=self.device)

        outputs = self.model(**inputs_data)

        label_preds = utils.normalized_prediction(outputs, self.tokenizer)
        data_dict['labels'] = label_preds[0]
        data = utils.clean_prediction_data(data_dict, self.tokenizer)
        data = utils.rebuild_prediction_data(data)
        
        return data



# data_dict = utils.annoset_transform(objects, tokenizer, max_seq_length=512)
# inputs_data = utils.annoset_inputs(data_dict, device=device)

# outputs = model(**inputs_data)

# label_preds = normalized_prediction(outputs, tokenizer)
# data_dict['labels'] = label_preds[0]
# data = clean_prediction_data(data_dict, tokenizer)
# data = rebuild_prediction_data(data)
# data
