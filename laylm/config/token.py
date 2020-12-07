import torch.nn as nn

cls_token_at_end=False
sep_token_extra=False
pad_on_left=False

cls_token = "[CLS]"
cls_token_id =  2

sep_token = "[SEP]"
sep_token_id = 3

pad_token = '[PAD]'
pad_token_id = 0

cls_token_segment_id=1
pad_token_segment_id=0
sequence_a_segment_id=0

cls_token_box = [0, 0, 0, 0]
sep_token_box = [1000, 1000, 1000, 1000]
pad_token_box = [0, 0, 0, 0]

pad_token_label_id = nn.CrossEntropyLoss().ignore_index
ignore_index_token_id = nn.CrossEntropyLoss().ignore_index
