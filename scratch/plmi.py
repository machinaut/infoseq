#!/usr/bin/env python
# %% trying to get power-law mutual information

import numpy as np

# %%
N = 4
# make a zipf distribution
power = 1.0
indexes = np.arange(1, N+1)
probs = indexes ** -power
probs = probs / np.sum(probs)
probs
# %% hierarchy
P = {i: np.roll(probs, -i) for i in range(2)}
for k, v in P.items():
    print(k, v)

# %% generate
result = [[0, 0], [0, 1], [1, 0], [1, 1]]
seq = [0]
def step(seq):
    new_seq = []
    for s in seq:
        draw = np.random.choice(np.arange(N), p=P[s])
        new_seq.extend(result[draw])
    return new_seq

for _ in range(10):
    seq = step(seq)
    print(seq)

