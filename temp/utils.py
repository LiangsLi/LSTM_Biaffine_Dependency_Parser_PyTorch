# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 17:05:02 2018

@author: mypc
"""
import torch
import torch.nn as nn
def seq_mask(seq_len, max_len):
    """Create sequence mask.
    :param seq_len: list or torch.Tensor, the lengths of sequences in a batch.
    :param max_len: int, the maximum sequence length in a batch.
    :return mask: torch.LongTensor, [batch_size, max_len]
    """
    if not isinstance(seq_len, torch.Tensor):
        seq_len = torch.LongTensor(seq_len)
    seq_len = seq_len.view(-1, 1).long()   # [batch_size, 1]
    seq_range = torch.arange(start=0, end=max_len, dtype=torch.long, device=seq_len.device).view(1, -1) # [1, max_len]
    return torch.gt(seq_len, seq_range) # [batch_size, max_len]




if __name__ == "__main__":
    seq_len = [2, 5, 7, 1]
    max_len = max(seq_len)
    batch_mask = seq_mask(seq_len, max_len)
    print(batch_mask.dtype)
    input_tensor = torch.randn(4, 7)
    print(input_tensor)
    input_tensor.data.masked_fill_(batch_mask==0, -float('inf'))
    softmax = nn.Softmax(dim=-1)
    output_tensor = softmax(input_tensor)
    print(input_tensor)
    print(output_tensor)
    