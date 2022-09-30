from lib2to3.pgen2.tokenize import tokenize
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from data_process import data_process

train_iter = WikiText2(split='train')
tokenize = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenize, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Data Processing
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)