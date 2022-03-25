#!/usr/bin/env python
# %%

# TODO: https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html


import argparse
import logging

import numpy as np
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# %%
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# %%
text = "Mary had a little lamb,\nits fleece was white as"
tokens = tokenizer.encode(text, return_tensors="pt")
logits = model(tokens)

# %%
generation = model.generate(tokens).squeeze()
tokenizer.decode(generation)

# %%
generator = pipeline('text-generation', model='gpt2')
generator(text)
# %%
type(generator.model)

# %% 
import torch

A = torch.rand(5, 7)
B = torch.rand(7, 11)
C = torch.rand(11, 13)

x = torch.tensor(torch.rand(5), requires_grad=True)

y = x @ A @ B @ C
torch.autograd.backward(y, retain_variables=True)
x.grad