#!/usr/bin/env python

'''
Here is our plan:

* Huggingface transformers GPT2 as a base
* Make a custom function, which just transports an input between two layers
    * Some function(A, B, output_A) - computes output layer B given output of layer A
* Compute the jacobian with torch.autograd.functional.jacobian
* Save these in a matrix of all As and Bs
* Plot this matrix for each token in a sequence: "The Eiffel Tower is in Paris"

Technical Requirements:
* Custom GPT2 class, or subclass of the huggingface GPT2, which adds this custom function
* Verify that we can give this custom function to the jacobian function

Skip for now:
* Building a proof-of-concept deep neural network
* Unit tests, or any kind of testing
'''

# %%
import torch
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

config = GPT2Config(output_hidden_states=True)
model = GPT2Model.from_pretrained('gpt2', config=config)

# %%

def make_requires_grad(module, input_):
    hidden, = input_
    print("make_requires_grad: input:", hidden.shape)
    result = hidden.clone().detach().requires_grad_(True)
    return (result,)

def create_fn_for_jacobian(layer_A, layer_B, tokens):
    ''' Creates the fun: layer_A_in -> layer_B_out '''
    assert 0 <= layer_A < layer_B <= 11, f"layer_A: {layer_A}, layer_B: {layer_B}"
    
    handle = model.h[layer_A].register_forward_pre_hook(make_requires_grad)

    def fn(layer_A_in):
        # compute and return layer_B output
        output = model(tokens)
        result =  output.hidden_states[layer_B + 1][0, -1, :]
        assert result.shape == (768,), f"result: {result.shape}"
        return result

    return fn, handle

def compute_jacobian(layer_A, layer_B, tokens):
    ''' Computes the jacobian for layer_A and layer_B '''
    fn, handle = create_fn_for_jacobian(layer_A, layer_B, tokens)
    output = model(tokens)
    layer_A_in = torch.tensor(output.hidden_states[layer_A].detach()[0, -1, :], requires_grad=True)
    assert layer_A_in.shape == (768,), f"layer_A_in: {layer_A_in.shape}"
    jac = torch.autograd.functional.jacobian(fn, layer_A_in)
    handle.remove()
    return jac

# Call the create_fn_for_jacobian to get a fn
tokens = tokenizer.encode("The Eiffel Tower is in Paris", return_tensors="pt")
jac = compute_jacobian(3, 8, tokens)
jac