#!/usr/bin/env python

# %% 
import random

class Tree:
    def __init__(self, token, children=None, seed=None):
        self.token = token
        self.children = children or {}
        self.rng = random.Random(seed)

    def add(self, token, text):
        t, *ext = text
        if t not in self.children:
            self.children[t] = Tree(token)
        if ext:
            self.children[t].add(token, ext)
    
    def encode_single(self, text, compression=.9):
        # Leaf node
        if not len(text):
            return self.token, b''
        t, *ext = text
        assert 0 <= t < 256, f't={t}'
        # Stochastically choose to leave the tree early
        if self.rng.random() > compression:
            return self.token, text
        # Otherwise, recurse to children if possible
        if t not in self.children:
            return self.token, text
        return self.children[t].encode_single(ext, compression=compression)

    def encode(self, text, compression=.9):
        tokens = []
        while len(text):
            token, text = self.encode_single(text, compression=compression)
            tokens.append(token)
        return tokens

    def decode(self, tokens):
        decode_map = self.decode_map
        return b''.join(decode_map[t] for t in tokens)

    def check(self, text):
        if not len(text):
            return True
        t = text[0]
        assert 0 <= t < 256, f't={t}'
        if t not in self.children:
            return False
        return self.children[t].check(text[1:])

    @property
    def encode_map(self):
        encode_map = {b'': self.token}
        for t, child in self.children.items():
            if child is not None:
                for text, token in child.encode_map.items():
                    encode_map[bytes([t]) + text] = token
        return encode_map

    @property
    def decode_map(self):
        return {v: k for k, v in self.encode_map.items()}

# %%
class Tokenization:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.tree = {}
        self.encode_map = {b'': 0}
        self.decode_map = {0: b''}

    def add(self, text: bytes):
        assert text, repr(text)
        assert text not in self.encode_map, f'{repr(text)} already in encode_map'
        current = self.tree
        # Add all substrings to the tree before the full string
        if text[:-1] not in self.encode_map:
            self.add(text[:-1])
        # Add new token to our maps
        token = len(self.encode_map)
        self.encode_map[text] = token
        self.decode_map[token] = text
        # Traverse the tree to add the node
        while text:
            t, *text = text
            if t not in current:
                current[t] = {}
            current = current[t]
        return token

    def encode_step(self, text: bytes, compression: float=0.9):
        ''' Encode a single step, returning the next token and remaining text '''
        assert 0.0 <= compression <= 1.0, repr(compression)
        # Traverse the tree
        code = []
        current = self.tree
        while text and text[0] in current and self.rng.random() < compression:
            code.append(text[0])
            current = current[text[0]]
            text = text[1:]
        # Return the token and remaining text
        return self.encode_map[bytes(code)], text

    def encode(self, text: bytes, compression: float=0.9):
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



# %%

class TokenizationOld:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.tree = Tree(0)
        self.decode_map = {0: b''}
        # Start with single bytes
        for i in range(256):
            self.add(bytes([i]))

    def encode(self, text, compression=.9):
        ''' Search forwards from shortest possible encoding '''
        return self.tree.encode(text, compression=compression)

    def decode(self, tokens):
        return b''.join(self.decode_map[t] for t in tokens)

    def add(self, text):
        ''' Add bytes as a token, and every left partial '''
        if len(text) > 1:
            self.add(text[:-1])
        if not self.tree.check(text):
            token = len(self.decode_map)
            assert token not in self.decode_map, f'token={token}'
            self.tree.add(token, text)
            self.decode_map[token] = text

# tok = Tokenization()
# tok.decode(tok.encode(b'abcd'))
