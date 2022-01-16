#!/usr/bin/env python
# %% get the unigram distribution for a dataset
import os
import torch
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import log_softmax

# TODO: wait til we download everything to do fancier stuff
data_path = '../data/frankenstein.txt'
data = open(data_path, 'rb').read()

len(data)
# %%
byte_count = defaultdict(int)
for byte in data:
    byte_count[byte] += 1

freq = np.array(sorted(byte_count.values(), reverse=True)) / len(data)
x = np.arange(len(freq)) + 1
source = torch.tensor(x, dtype=float)
target = torch.tensor(freq, dtype=float)

def get_xent(params, fn, target):
    return (-log_softmax(fn(params)) * target).sum()

def fit_xent(params, fn, target):
    opt = torch.optim.Adam(params, lr=0.01)
    for _ in range(1000):
        opt.zero_grad()
        get_xent(params, fn, target).backward()
        opt.step()

def get_pred(params, fn):
    return torch.softmax(fn(params), dim=0)

# fit a simple exponential law: P(b) \propto exp(-b * x)
b = torch.tensor(0.1, requires_grad=True)
params = [b]
exp_fn = lambda params: -(params[0]) * source
fit_xent(params, exp_fn, target)
with torch.no_grad():
    y = get_pred(params, exp_fn)
    l = get_xent(params, exp_fn, target)
print('b', b.item(), 'loss', l.item())

latex = f'$\log P(b) \propto -{b.item():0.3f}x$'

fig, ax = plt.subplots()
ax.plot(x, freq, label='data')
ax.plot(x, y, label=latex)
ax.set_yscale('log')
ax.set_xlabel('byte index (sorted by frequency)')
ax.set_ylabel('frequency (log scale)')
ax.legend()
fig.suptitle('Byte frequency distribution, with exponential fit')
fig.tight_layout()

# %%
bigram = defaultdict(int)
for i in range(len(data) - 1):
    bigram[data[i], data[i + 1]] += 1

freq = np.array(sorted(bigram.values(), reverse=True)) / (len(data) - 1)
x = np.arange(len(freq))
source = torch.tensor(x, dtype=float)
target = torch.tensor(freq, dtype=float)
# fit a simple exponential law: P(b) \propto exp(-b * x)
b = torch.tensor(0.1, requires_grad=True)
params = [b]
exp_fn = lambda params: -(params[0]) * source
fit_xent(params, exp_fn, target)
y = get_pred(params, exp_fn)
print('b', b.item())

latex = f'$\log P(b) \propto -{b.item():0.3f}x$'

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, freq, label='data')
ax.plot(x, y, label=latex)
ax.set_yscale('log')
ax.set_xlabel('Bigram index (sorted by frequency)')
ax.set_ylabel('frequency (log scale)')
ax.legend()
fig.suptitle('Bigram frequency distribution, with exponential fit')
fig.tight_layout()

# %% byte pair encoding

