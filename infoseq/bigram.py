#!/usr/bin/env python
# %% plot bigram of kjv

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# %%
data_path = '../data/kjv.txt'
data = open(data_path, 'rb').read()

# %%
unigrams = defaultdict(int)
for byte in data:
    unigrams[byte] += 1

counts = sorted(unigrams.values(), reverse=True)
fig, ax = plt.subplots()
ax.plot(np.arange(len(counts)) + 1, counts)
ax.set_xscale('log')
ax.set_yscale('log')

# %%
bigrams = defaultdict(int)
for i in range(len(data) - 1):
    bigrams[(data[i], data[i+1])] += 1

counts = sorted(bigrams.values(), reverse=True)
fig, ax = plt.subplots()
ax.plot(np.arange(len(counts)) + 1, counts)
ax.set_xscale('log')
ax.set_yscale('log')

# %% top 10 bigrams
top_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)[:10]
for (a, b), count in top_bigrams:
    print(bytes([a,b]), ':', count)