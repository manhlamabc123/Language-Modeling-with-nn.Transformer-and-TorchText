from torch.utils.data import dataset
from torch import nn, Tensor, numel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def data_process(raw_text_iter: dataset.IterableDataset, vocab, tokenizer) -> Tensor:
    # Converts raw text into a flat Tensor
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel > 0, data)))