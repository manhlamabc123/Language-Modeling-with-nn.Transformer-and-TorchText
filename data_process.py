from torch.utils.data import dataset
from torch import nn, Tensor, numel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from constants import DEVICE

def data_process(raw_text_iter: dataset.IterableDataset, vocab, tokenizer) -> Tensor:
    # Converts raw text into a flat Tensor
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel > 0, data)))

def batchify(data: Tensor, bsz: int) -> Tensor:
    '''
    Divides the data into bsz separate sequences, removing extra elements the wouldn't cleanly fit

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size
    Returns:
        Tensor of shape [N // bsz, bsz]
    '''
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(DEVICE)