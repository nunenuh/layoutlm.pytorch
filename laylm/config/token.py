import torch.nn as nn

cls_token_at_end=False
cls_token="[CLS]"
sep_token="[SEP]"
sep_token_extra=False

cls_token_segment_id=1
pad_token_segment_id=0
sequence_a_segment_id=0

pad_on_left=False
pad_token=0

cls_token_box=[0, 0, 0, 0]
sep_token_box=[1500, 1500, 1500, 1500]
pad_token_box=[0, 0, 0, 0]

pad_token_label_id = nn.CrossEntropyLoss().ignore_index
