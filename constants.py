import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpy')
BATCH_SIZE = 20
EVAL_BATCH_SIZE = 10
BPTT = 35
EMBEDDING_SIZE = 200
D_HID = 200
N_LAYERS = 2
N_HEAD = 2
DROPOUT = 0.2
BEST_VAL_LOSS = float('inf')
EPOCHS = 3
best_model = None
LEARNING_RATE = 5.0