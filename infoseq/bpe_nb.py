#!/usr/bin/env python
# %% byte pair encoding
import os
from collections import defaultdict
import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from torch.nn.functional import log_softmax

# TODO: wait til we download everything to do fancier stuff
data_path = '../data/kjv.txt'
data = open(data_path, 'r').read().encode()

chunk_size = 10_000
chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

len(data), len(chunks)
# %%
def tokenize(data, encode):
    tokens = []
    longest = max(len(k) for k in encode.keys())
    i = 0
    while i < len(data):
        for j in range(longest, 0, -1):
            if i + j <= len(data):
                seg = data[i:i+j]
                # print('checking seg', seg)
                if seg in encode:
                    # print('adding seg', seg)
                    tokens.append(encode[seg])
                    i += j
                    break
        else:
            assert False, f'could not tokenize {data[i:i+10]}'
    return tokens

def get_pairs(tokens):
    pairs = defaultdict(int)
    for i in range(len(tokens) - 1):
        pairs[(tokens[i], tokens[i+1])] += 1
    return dict(pairs)

def get_freq(tokens):
    count = defaultdict(int)
    for t in tokens:
        count[t] += 1
    freq = np.array(sorted(count.values(), reverse=True))
    return freq

# %%
decode = {i: bytes([b])for i, b in enumerate(set(data))}
encode = {v: k for k, v in decode.items()}
saved_freqs = {len(encode): get_freq(tokenize(data, encode))}

# %%
for i in range(100):
    tokens = tokenize(data, encode)
    pairs = get_pairs(tokens)
    if max(pairs.values()) < 2:
        break
    # TODO: randomly break ties
    a, b = max(pairs, key=pairs.get)
    ab = decode[a] + decode[b]
    if i % 10 == 0:
        print(len(decode), ':', a, b, ab)
    encode[ab] = len(decode)
    decode[len(decode)] = ab
    saved_freqs[len(encode)] = get_freq(tokens)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
norm = colors.Normalize(vmin=min(saved_freqs), vmax=max(saved_freqs))
for i, freq in saved_freqs.items():
    c = plt.colormaps['viridis'](norm(i))
    ax.plot(freq, c='k' if i % 100 == 0 else c)
ax.set_yscale('log')
ax.set_ylabel('token frequency')
ax.set_xlabel('token index (sorted by frequency)')
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), label='encoding size (in tokens)')
fig.suptitle('Byte Pair Encoding effect on token frequency')
fig.tight_layout()

# %% Plot the entropy of the unigram distributions
fig, ax = plt.subplots()
keys = sorted(saved_freqs.keys())
sums = {k: saved_freqs[k].sum() for k in keys}
normed = {k: saved_freqs[k] / sums[k] for k in keys}
entropies = {k: (-normed[k] * np.log(normed[k])).sum() for k in keys}
ax.plot(keys, [entropies[k] for k in keys])
ax.set_xlabel('encoding size (in tokens)')
ax.set_ylabel('unigram entropy (in nats)')
# ax.set_xscale('log')
# ax.set_yscale('log')
fig.suptitle('Byte Pair Encoding effect on entropy')
fig.tight_layout()

# %% Plot the entropy of the unigram distributions
# fig, ax = plt.subplots()
# keys = sorted(saved_freqs.keys())
# sums = {k: saved_freqs[k].sum() for k in keys}
# normed = {k: saved_freqs[k] / sums[k] for k in keys}
# entropies = {k: (-normed[k] * np.ones_like(normed[k])).sum() for k in keys}
# ax.plot(keys, [entropies[k] for k in keys])
# ax.set_xlabel('encoding size (in tokens)')
# ax.set_ylabel('unigram entropy (in nats)')
# ax.set_scale
# fig.suptitle('Byte Pair Encoding effect on entropy')
# fig.tight_layout()
encode.keys()