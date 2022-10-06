import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpy')
BATCH_SIZE = 20
EVAL_BATCH_SIZE = 10
BPTT = 35