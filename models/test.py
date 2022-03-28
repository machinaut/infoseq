#!/usr/bin/env python
# %%

import torch

x = torch.tensor([[1.,2.,3.]], requires_grad=True)
A = torch.tensor([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 0, 0]])
f = lambda x: torch.mm(x, A)
torch.autograd.functional.jacobian(f, x)
