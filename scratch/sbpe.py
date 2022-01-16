#!/usr/bin/env python
# %% Stochastic Byte Pair Encoding stats

import torch

alpha = torch.tensor(.01, requires_grad=True, dtype=float)
beta = torch.tensor(2/3 + .03, requires_grad=True, dtype=float)
delta = torch.tensor(.123, requires_grad=True, dtype=float)
gamma = torch.tensor(1/3 + .02, requires_grad=True, dtype=float)
eta = torch.tensor(.991, requires_grad=True, dtype=float)

params = [alpha, beta, delta, gamma, eta]
optim = torch.optim.Adam(params, lr=0.01)


curve = []
for _ in range(1000):
    optim.zero_grad()
    loss = torch.tensor(0.0, dtype=float)
    Pa = (1 - alpha) * (1 - beta)
    loss += (Pa - 1/6) ** 2
    Pab = alpha * delta + (1 - alpha) * beta * gamma
    loss += (Pab - 5/18) ** 2
    Pabc = alpha * (1 - delta) + (1 - alpha) * beta * (1 - gamma)
    loss += (Pabc - 5/9) ** 2
    Palc = eta * (1 - beta)
    loss += (Palc - 1/24) ** 2
    Pablc = eta * beta + (1 - eta)
    loss += (Pablc - 5/24) ** 2
    curve.append(loss.detach().item())
    loss.backward()
    optim.step()

import matplotlib.pyplot as plt

plt.plot(curve)
curve[-1]