from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import datetime
timestamp_now=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# constants

NUM_BATCHES = int(2e2)
HASHES = 2
BATCH_SIZE = 2
GRADIENT_ACCUMULATE_EVERY = 2
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 2
GENERATE_EVERY  = 10
GENERATE_LENGTH = 512
SEQ_LEN = 4096
loss_list=[]
# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def save_plot(list):
    fig=plt.figure()
    x=[i*VALIDATE_EVERY for i in range(len(list))]
    plt.plot(x,list)
    fig.savefig('../../../saved_figures/loss_graph_hash={}_{}.png'.format(HASHES,timestamp_now))
# instantiate model

model = ReformerLM(
    dim = 128,
    depth = 24,
    max_seq_len = SEQ_LEN,
    num_tokens = 256,
    heads = 16,
    bucket_size = 64,
    n_hashes = HASHES,
    ff_chunks = 10,
    lsh_dropout = 0.1,
    weight_tie = True,
    causal = True,
    n_local_attn_heads = 4,
    use_full_attn = False # set this to true for comparison with full attention
)

model = TrainingWrapper(model)
model.cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader), return_loss = True)
        loss.backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader), return_loss = True)
            print(f'validation loss: {loss.item()}')
            loss_list.append(loss.item())

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(inp, GENERATE_LENGTH)
        output_str = decode_tokens(sample)
        print(output_str)
save_plot(loss_list)
