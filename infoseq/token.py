#!/usr/bin/env python
# %%
import random


class Tokenization:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.tree = {}
        self.encode_map = {b'': 0}
        self.decode_map = {0: b''}

    def add(self, text: bytes):
        ''' Add a new code to the tree, returning the token '''
        assert text, repr(text)
        assert text not in self.encode_map, f'{repr(text)} already in encode_map'
        current = self.tree
        # Add all substrings to the tree before the full string
        if text[:-1] not in self.encode_map:
            self.add(text[:-1])
        # Add new token to our maps
        token = len(self.encode_map)
        assert token not in self.decode_map, f'{token} already in decode_map'
        self.encode_map[text] = token
        self.decode_map[token] = text
        # Traverse the tree to add the node
        while text:
            t, *text = text
            if t not in current:
                current[t] = {}
            current = current[t]
        return token

    def encode_step(self, text: bytes, compression: float = 0.9):
        ''' Encode a single step, returning the next token and remaining text '''
        assert 0.0 <= compression <= 1.0, repr(compression)
        # Traverse the tree
        code = []
        current = self.tree
        while text and text[0] in current and self.rng.random() < compression:
            code.append(text[0])
            current, text = current[text[0]], text[1:]
        # Return the token and remaining text
        return self.encode_map[bytes(code)], text

    def encode(self, text: bytes, compression: float = 0.9):
        ''' Encode a string, returning a list of tokens '''
        assert 0.0 <= compression <= 1.0, repr(compression)
        # Encode the text step by step
        tokens = []
        while len(text):
            token, text = self.encode_step(text, compression=compression)
            tokens.append(token)
        return tokens

    def decode(self, tokens: list):
        ''' Decode a list of tokens, returning the original text '''
        return b''.join(self.decode_map[t] for t in tokens)
