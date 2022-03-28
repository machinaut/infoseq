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

handles_to_remove = []


def make_injection_function(layer_A_input):
    def fn(module, input_):
        ''' Injects layer_A_in into the module '''
        hidden, = input_
        # make a detached cloned copy of hidden
        hidden_clone = hidden.clone().detach()
        # drop the last token position from it
        hidden_clone = hidden_clone[:, :-1, :]
        # concatenate it with layer_A_in
        layer_A_unsqueezed = layer_A_input.unsqueeze(0).unsqueeze(0)
        result = torch.cat((hidden_clone, layer_A_unsqueezed), dim=1)
        # Assert layer_A_input requires_grad
        assert layer_A_input.requires_grad, f"layer_A_input: {layer_A_input.shape}"
        # Assert that result requires_grad
        assert result.requires_grad, f"result: {result.shape}"
        # Assert result is same shape as hidden
        assert result.shape == hidden.shape, f"layer_A_input: {layer_A_input.shape}, hidden: {hidden.shape}"
        return (result,)
    return fn


def create_fn_for_jacobian(layer_A, layer_B, tokens):
    ''' Creates the fun: layer_A_in -> layer_B_out '''
    assert 0 <= layer_A < layer_B <= 11, f"layer_A: {layer_A}, layer_B: {layer_B}"
    
    def fn(layer_A_in):
        global handles_to_remove
        # compute and return layer_B output
        hook = make_injection_function(layer_A_in)
        handle = model.h[layer_A].register_forward_pre_hook(hook)
        handles_to_remove.append(handle)

        output = model(tokens)
        result =  output.hidden_states[layer_B + 1][0, -1, :]
        assert result.shape == (768,), f"result: {result.shape}"
        return result

    return fn

def compute_jacobian(layer_A, layer_B, tokens):
    ''' Computes the jacobian for layer_A and layer_B '''
    global handles_to_remove
    fn = create_fn_for_jacobian(layer_A, layer_B, tokens)
    output = model(tokens)
    layer_A_in = output.hidden_states[layer_A].clone().detach()[0, -1, :]
    assert layer_A_in.shape == (768,), f"layer_A_in: {layer_A_in.shape}"
    jac = torch.autograd.functional.jacobian(fn, layer_A_in)
    while handles_to_remove:
        handles_to_remove.pop().remove()
    return jac

# Call the create_fn_for_jacobian to get a fn
tokens = tokenizer.encode("The Eiffel Tower is in Paris", return_tensors="pt")
jac = compute_jacobian(3, 8, tokens)
jac

# %%
import matplotlib.pyplot as plt

plt.matshow(jac)